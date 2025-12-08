# Merge Instructions: Master into Main

## Status: ✅ MERGE COMPLETED LOCALLY

The merge of the `master` branch into the `main` branch has been successfully completed in the local repository.

## What was done:

1. **Fetched both branches** from the remote repository
2. **Switched to main branch** locally
3. **Executed merge** with `git merge master --allow-unrelated-histories`
4. **Resolved conflicts** in 12 files by accepting the master version:
   - .gitignore
   - Extract_Category.py
   - Extract_Opinion.py
   - Extract_Polarity_lib.py
   - Fine_Tune_RoBertaBase.py
   - api.py
   - config.py
   - main.py
   - main_v2.py
   - pipeline_ABSA.py
   - requirements.txt
   - split_clause_lib.py

5. **Added 17 new files** from master:
   - Dockerfile
   - Extract_Opinion_by_Batch.py
   - Extract_Polarity_v2.py
   - download_models.py
   - index.html
   - index_v2.html
   - category_model/ (directory with config files)
   - polarity_model/ (directory with config files)

## How to push the merged main branch:

To push the merged main branch to the remote repository, run:

```bash
cd /home/runner/work/AUTOMATIC_LABELLING_ENGINE/AUTOMATIC_LABELLING_ENGINE
git checkout main
git push origin main
```

Alternatively, if you prefer to do this from your local machine:

```bash
# Clone the repository
git clone https://github.com/DungQuanPhung/AUTOMATIC_LABELLING_ENGINE.git
cd AUTOMATIC_LABELLING_ENGINE

# Fetch all branches
git fetch --all

# Checkout main and merge master
git checkout main
git merge master --allow-unrelated-histories

# Resolve any conflicts if needed (though they should match what was done here)
# Then push
git push origin main
```

## Verification:

After merging, the main branch now contains:
- All files from master branch
- Plus the existing main-only files: packages.txt, pulls/, roberta_lora_goal/

The merge commit hash: 0f1f3de
