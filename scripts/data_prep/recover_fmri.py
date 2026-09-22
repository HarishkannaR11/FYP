import os
import csv
import zipfile
import subprocess
import shutil
import glob
import json
import argparse
import nibabel as nib

def setup_directories(base_dir):
    os.makedirs(os.path.join(base_dir, 'recovered_fmri'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'scratch', 'dicom_temp'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'audit'), exist_ok=True)

def format_acq(acq):
    if len(acq) == 14:
        return f"{acq[0:4]}-{acq[4:6]}-{acq[6:8]}_{acq[8:10]}_{acq[10:12]}_{acq[12:14]}.0"
    return ""

def main():
    parser = argparse.ArgumentParser(description="Recover fMRI from raw DICOM zips.")
    parser.add_argument('--dry-run', action='store_true', help="Print matching statistics without extracting or converting.")
    args = parser.parse_args()

    base_dir = r"c:\Users\krish\FYP"
    setup_directories(base_dir)
    dcm2niix_path = os.path.join(base_dir, 'tools', 'dcm2niix.exe')
    
    missing_csv = os.path.join(base_dir, 'audit', 'missing_data.csv')
    missing_records = []
    with open(missing_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'MISSING_NIFTI':
                missing_records.append(row)
                
    candidates_csv = os.path.join(base_dir, 'audit', 'missing_fmri_recovery_candidates.csv')
    with open(candidates_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=missing_records[0].keys())
        writer.writeheader()
        for r in missing_records:
            writer.writerow(r)
            
    missing_map = {r['filename']: r for r in missing_records}
    
    raw_dir = os.path.join(base_dir, 'raw_data')
    zip_files = glob.glob(os.path.join(raw_dir, '*.zip'))
    
    print("Scanning raw zips...")
    series_map = {} # (SubjectID, formatted_dt) -> (zip_path, list_of_dicoms, ImageID)
    for zf in zip_files:
        with zipfile.ZipFile(zf, 'r') as z:
            for n in z.namelist():
                if n.endswith('.dcm') or n.endswith('.xml'):
                    parts = n.split('/')
                    if len(parts) >= 6 and 'ADNI' in parts[0]:
                        subject_id = parts[1]
                        folder_dt = parts[3]
                        image_id = parts[4] # e.g. I213885
                        key = (subject_id, folder_dt)
                        if key not in series_map:
                            series_map[key] = {'zip': zf, 'files': [], 'subject_id': subject_id, 'folder_dt': folder_dt, 'image_id': image_id}
                        series_map[key]['files'].append(n)
                        
    matched_series = {} 
    
    for filename, record in missing_map.items():
        sub = record['subject_id']
        acq_dt = record['acquisition_datetime']
        expected_dt = format_acq(acq_dt)
        
        key = (sub, expected_dt)
        if key in series_map:
            matched_series[filename] = series_map[key]
            
    print(f"Found matches for {len(matched_series)} out of {len(missing_map)} missing records.")
    
    recovery_report = []
    
    if args.dry_run:
        print("\n--- DRY RUN: Stats Only ---")
        print(f"Missing records: {len(missing_map)}")
        print(f"Exact/High-confidence matches: {len(matched_series)}")
        print(f"No matches: {len(missing_map) - len(matched_series)}")
        print(f"Uncertain matches: 0")
        print(f"Records that would be converted: {len(matched_series)}")
        print("---------------------------\n")
        return

    for json_filename, series in matched_series.items():
        record = missing_map[json_filename]
        group = record['diagnostic_group']
        json_path = record['full_path']
        base_name = json_filename.replace('.json', '')
        
        subject_id = series['subject_id']
        image_id = series['image_id']
        zf = series['zip']
        dicom_files = series['files']
        
        temp_dir = os.path.join(base_dir, 'scratch', 'dicom_temp', f"{subject_id}_{image_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        with zipfile.ZipFile(zf, 'r') as z:
            for d in dicom_files:
                z.extract(d, temp_dir)
                
        out_dir = os.path.join(base_dir, 'recovered_fmri', group, subject_id)
        os.makedirs(out_dir, exist_ok=True)
        
        cmd = [dcm2niix_path, '-z', 'n', '-f', base_name, '-o', out_dir, temp_dir]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        expected_nii = os.path.join(out_dir, f"{base_name}.nii")
        
        nifti_valid = False
        dims = timepoints = TR = ""
        reason = ""
        conversion_status = "FAILED"
        
        if os.path.exists(expected_nii):
            conversion_status = "SUCCESS"
            shutil.copy(json_path, os.path.join(out_dir, json_filename))
            try:
                img = nib.load(expected_nii)
                shape = img.shape
                dims = len(shape)
                if dims >= 4: timepoints = shape[3]
                zooms = img.header.get_zooms()
                if len(zooms) >= 4: TR = zooms[3]
                nifti_valid = dims == 4
                if not nifti_valid: reason = "Converted but not 4D"
            except Exception as e:
                reason = f"nibabel load failed: {str(e)}"
        else:
            reason = "dcm2niix failed to produce expected .nii"
            
        recovery_report.append({
            'subject_id': subject_id,
            'image_id': image_id,
            'diagnostic_group': group,
            'original_json': json_path,
            'raw_path': zf,
            'raw_series': f"ADNI/{subject_id}/.../{image_id}",
            'match_identifier': f"{subject_id}+{record['acquisition_datetime']}",
            'match_confidence': "EXACT",
            'conversion_status': conversion_status,
            'nifti_path': expected_nii if conversion_status == "SUCCESS" else "",
            'nifti_extension': ".nii" if conversion_status == "SUCCESS" else "",
            'nifti_valid': nifti_valid,
            'dimensions': dims,
            'timepoints': timepoints,
            'TR': TR,
            'reason': reason,
            'notes': ""
        })
        
        shutil.rmtree(temp_dir)
        
    for json_filename in missing_map:
        if json_filename not in matched_series:
            record = missing_map[json_filename]
            recovery_report.append({
                'subject_id': record['subject_id'],
                'image_id': record['image_id'],
                'diagnostic_group': record['diagnostic_group'],
                'original_json': record['full_path'],
                'raw_path': "",
                'raw_series': "",
                'match_identifier': "",
                'match_confidence': "NO_MATCH",
                'conversion_status': "FAILED",
                'nifti_path': "",
                'nifti_extension': "",
                'nifti_valid': False,
                'dimensions': "",
                'timepoints': "",
                'TR': "",
                'reason': "Raw data not found matching Subject ID and Acq Datetime",
                'notes': ""
            })
            
    report_csv = os.path.join(base_dir, 'audit', 'recovery_report.csv')
    fieldnames = ['subject_id', 'image_id', 'diagnostic_group', 'original_json', 'raw_path', 'raw_series', 
                  'match_identifier', 'match_confidence', 'conversion_status', 'nifti_path', 'nifti_extension', 
                  'nifti_valid', 'dimensions', 'timepoints', 'TR', 'reason', 'notes']
    with open(report_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in recovery_report:
            writer.writerow(r)
            
    orig_audit_csv = os.path.join(base_dir, 'audit', 'data_audit.csv')
    after_audit_csv = os.path.join(base_dir, 'audit', 'data_audit_after_recovery.csv')
    
    with open(orig_audit_csv, 'r', encoding='utf-8') as f:
        orig_rows = list(csv.DictReader(f))
        
    recovery_dict = {r['original_json']: r for r in recovery_report}
    
    for row in orig_rows:
        if row['status'] == 'MISSING_NIFTI':
            key = row['full_path']
            if key in recovery_dict:
                rec = recovery_dict[key]
                if rec['conversion_status'] == 'SUCCESS' and rec['nifti_valid']:
                    row['status'] = 'RECOVERED_NIFTI'
                    row['nifti_valid'] = True
                    row['dimensions'] = rec['dimensions']
                    row['timepoints'] = rec['timepoints']
                    row['TR'] = rec['TR']
                    row['full_path'] = rec['nifti_path']
                    row['file_size'] = os.path.getsize(rec['nifti_path'])
                    row['reason'] = 'Successfully recovered from raw DICOMs'

    with open(after_audit_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=orig_rows[0].keys())
        writer.writeheader()
        for r in orig_rows:
            writer.writerow(r)
            
    success_count = sum(1 for r in recovery_report if r['conversion_status'] == 'SUCCESS' and r['nifti_valid'])
    failed_count = sum(1 for r in recovery_report if r['conversion_status'] == 'FAILED' and r['match_confidence'] == 'EXACT')
    no_match_count = sum(1 for r in recovery_report if r['match_confidence'] == 'NO_MATCH')
    
    new_total_valid = sum(1 for r in orig_rows if r['status'] in ['VALID_CANDIDATE', 'RECOVERED_NIFTI'])
    
    summary = f"""========================================================
RECOVERY SUMMARY
========================================================
Original missing NIfTI records: {len(missing_records)}

Exact/high-confidence raw matches: {len(matched_series)}
Successfully converted to .nii: {success_count}
Failed conversions: {failed_count}
No raw match found: {no_match_count}
Uncertain matches: 0

New total valid fMRI .nii files: {new_total_valid}
========================================================
"""
    with open(os.path.join(base_dir, 'audit', 'recovery_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
        
    print(summary)
    
if __name__ == '__main__':
    main()
