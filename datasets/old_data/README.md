# Legacy dataset

`original_with_pca.csv` is the older PCA dataset used by scripts in
`oldScripts/`. It does not match the schema of the current PCA datasets in
`datasets/`.

Key differences:

- PCA columns are named `PC1_Bottom`, `PC1_InnerShape`, `PC1_OuterShape` (plus
  PC2/PC3 variants).
- Raw geometry samples use the older `inner_*`, `outer_*`, `innerShape_*`,
  `outerShape_*` naming.

If you run legacy scripts, use this file or update the scripts to the new
schema.
