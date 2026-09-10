# Face Recognition Project

## Prepare Evaluation Images

Ground-truth labels must be supplied manually. The preparation tool never uses face recognition to decide a label.

For images of a registered person, place source images in any folder and run:

```powershell
venv\Scripts\python.exe scripts\prepare_evaluation_dataset.py --source "C:\path\to\varshini_images" --kind known --person-name "Varshini"
```

The images are copied to `data/evaluation/known/Varshini/`. The original files are not modified.

For people who are not registered, run:

```powershell
venv\Scripts\python.exe scripts\prepare_evaluation_dataset.py --source "C:\path\to\unknown_images" --kind unknown
```

The images are copied to `data/evaluation/unknown/`. Use the `--help` option to see all arguments:

```powershell
venv\Scripts\python.exe scripts\prepare_evaluation_dataset.py --help
```

Only readable common image formats are copied. Existing destination filenames are preserved and new copies receive a suffix instead of overwriting an earlier file.

## Selecting Images Safely

For a large folder such as OneDrive Camera Roll, first create an inventory. This does not copy images or assign labels:

```powershell
venv\Scripts\python.exe scripts\inventory_images.py --source "C:\Users\varsh\OneDrive\Pictures" --source "C:\Users\varsh\OneDrive\Pictures\Camera Roll" --output "data\evaluation\pictures_inventory.csv"
```

Open the CSV, manually choose the correct full paths, and repeat `--file` once for each selected image. For example:

```powershell
venv\Scripts\python.exe scripts\prepare_evaluation_dataset.py --source "C:\Users\varsh\OneDrive\Pictures" --kind known --person-name "Varshini" --file "C:\Users\varsh\OneDrive\Pictures\image1.jpg" --file "C:\Users\varsh\OneDrive\Pictures\Camera Roll\image2.jpg"
```

For another known person, change `--person-name`. For manually selected unknown images, use:

```powershell
venv\Scripts\python.exe scripts\prepare_evaluation_dataset.py --source "C:\Users\varsh\OneDrive\Pictures" --kind unknown --file "C:\Users\varsh\OneDrive\Pictures\Camera Roll\unknown1.jpg"
```

Never assume that all images in a source folder share one identity. Review each selected path and label yourself before running evaluation.

## Run Evaluation

After checking that every folder label is correct, run:

```powershell
venv\Scripts\python.exe scripts\evaluate_system.py --dataset data\evaluation
```

The evaluator writes reports under `data/evaluation/`. Accuracy results are meaningful only when the dataset labels are correct and the images are separate from registration images.