for m in nipype ants nibabel nilearn numpy scipy templateflow; do
  python3 -c "import \; print('OK ', '\', getattr(\,'__version__','?'))" 2>/dev/null || echo "MISSING  \"
done