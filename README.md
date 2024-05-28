# face-detection-exercise

Face detection and recognition using OpenCV and Dlib and also calculating the False Acceptance Rate (FAR), False Rejection Rate (FRR), False Match Rate (FMR), False Non-Match Rate (FNMR) and accuracy from video data.

## What you need

- Python 3.8 or lower
- Installation of python requirements
- A dataset of faces for reference and stranger images
- A base image from you
- A video file or a live video stream

### Installing the requirements
```
pip install -r requirements.txt
```
### Download the dataset
Download the dataset for faces from [here](https://sites.google.com/view/sof-dataset) and place it in the root directory under `/unknown_faces` with following commands:
```bash
curl -L -o unknown_faces.rar https://drive.usercontent.google.com/download?id=1ufydwhMYtOhxgQuHs9SjERnkX0fXxorO&export=download&authuser=0&confirm=t&uuid=91495fca-e7bd-48bf-a911-ff07c2646ddc&at=APZUnTU3Uyl0y9aioFBFdWB9V_zb%3A1716813980194
unrar x unknown_faces.rar unknown_faces/
```

## Usage
### 1. Extract faces from your images
```python
python3 extract_faces.py --images_path <path_to_images> --output_path <path_to_save_extracted_faces> --reference_image <path_to_reference_image>
```



### 2. Calculate the optimal threshold
```python
python3 calculate_threshold.py --reference_images <path_to_reference_images> --stranger_images <path_to_stranger_images> --base_image <path_to_base_image>
```


### 3. Face detection and recognition
#### For recorded videos
```python
python3 main.py --video_path <path_to_video>
```
#### For live videos
```python
python3 main.py --live --reference_images <path_to_reference_images>
```

## How it is working
**Calculation**

![How it is working](https://github.com/vabulus/face-detection-exercise/blob/master/decision.png?raw=true)

**Application logic**

![Application Logic](https://github.com/vabulus/face-detection-exercise/blob/master/application-logic.png?raw=true)

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