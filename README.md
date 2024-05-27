# face-detection-exercise

Face detection and recognition using OpenCV and Dlib and also calculating the False Acceptance Rate (FAR), False Rejection Rate (FRR), False Match Rate (FMR), False Non-Match Rate (FNMR) and accuracy from video data.

## Installation

Tested on Python 3.8!

```
pip install -r requirements.txt
```

Download the dataset for faces from [here](https://sites.google.com/view/sof-dataset) and place it in the root directory under `/unknown_faces` with following commands:

```bash
curl -L -o unknown_faces.rar https://drive.usercontent.google.com/download?id=1ufydwhMYtOhxgQuHs9SjERnkX0fXxorO&export=download&authuser=0&confirm=t&uuid=91495fca-e7bd-48bf-a911-ff07c2646ddc&at=APZUnTU3Uyl0y9aioFBFdWB9V_zb%3A1716813980194
unrar x unknown_faces.rar unknown_faces/
```

## Usage

### Extract faces from images

Synthax

```python
python3 extract_faces.py --images_path <path_to_images> --output_path <path_to_save_extracted_faces> --reference_image <path_to_reference_image>
```

Example

```
python3 extract_faces.py --images_path images/ --output_path extracted_faces/ --reference_image example_face.jpg
```

### Calculate FAR/FRR, FMR/FNMR, accuracy

#### For recorded videos

```python
python3 main.py --video_path <path_to_video>
```

#### For live videos

```python
python3 main.py --live --reference_images <path_to_reference_images>
```

## Tips
Rename files in a directory with a name and a counter:
```
#!/bin/bash
counter=1
for file in *; do
  if [ -f "$file" ]; then
    mv "$file" "f${counter}.jpg"
    ((counter++))
  fi
done
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

