# Speaker notes: the biomarkers, in simple words

For slides 8 ("Every Scan in Every Stage") and 9 ("The Four Functional Biomarkers").
About 1.5 minutes. Say it slowly; point at each picture as you go.

## Slide 8 – every scan in every stage (15 s)

"These are all the scans we use. We have about 30 scans in each of the six stages –
between 21 and 32 depending on the stage – and 165 in total, from 127 people.
Each small picture is one scan. The one with the red box has a technical artefact; we
report it openly and keep it out of the quality summaries."

## Slide 9 – the four biomarkers (60–75 s)

"While the person rests in the scanner, we record the brain every 3 seconds. In every
small region of the brain the signal goes up and down slowly, like a wave. We ask four
simple questions about that wave."

1. **ALFF – how strong?** "ALFF measures how big the swings of the wave are in a region.
   Big swings mean stronger activity."
2. **ReHo – is the region working in step?** "ReHo looks at the small spots inside one
   region. If they rise and fall together, the region is working in step."
3. **FC – who talks to whom?** "Functional connectivity compares two different regions.
   If their waves rise and fall together, we call them connected. Doing this for every
   pair of our 246 regions gives a 246-by-246 table."
4. **DC – how connected is a region overall?** "Degree centrality just adds up all the
   connections of one region. A big total means the region is a hub."

"So every scan gives each region three numbers – strength, in-step-ness and total
connections – plus the full table of who is connected to whom. We keep all four because
each one shows a different side of the brain, and our model will use them together."

## If the panel asks

- **Why four and not one?** Earlier work mostly uses one at a time; each captures
  something different, and combining them is the point of the project.
- **Is DC the same as FC?** DC is built from FC: it is the row total. We checked that the
  "FC strength" we planned equals DC divided by 245 (to rounding error), so we kept DC and FC and
  dropped the duplicate.
- **Do the biomarkers already separate the stages?** Not in simple group averages (our
  Finding 2). That is why a combined, multivariate model is needed.
- **Are these 30 scans per stage all different people?** No. 165 scans come from 127
  people, because some were scanned more than once; the smallest stage (MCI) has 14 people.
