1. Train a large model for each method and compare their performance.
    ART, Diffusion, SET-GUIDE
2. (done) Make sure each model works for the aggregated hourly data.
3. (done) Make a script to transfer the hourly data into usable parquet files on the hard drive.
4. (done) Make an argument in parquet loader that will output an aggregate stream alongside the other phases.
5. Make a secondary model that is capable of splitting aggregation into phases.
6. Make a model that will do both simultaniously.
7. Test all possible models. (with same amount of parameters.)
    Phased: ART, Diffusion, SET-GUIDE
    GUIDE+SPLITTER, ART+SPLITTER, Diffusion+SPLITTER
    Disag-Guide, Disag-ART, Disag-Diffusion
8. Test different conditions and their effects.