# Image Processing Task for DKNSB Medical Device Consultants

## Task Description

The assigned task was to extract important information from a given set of images. Every image has the same details at the same specific areas. This information would need to be extracted from the image, and saved onto an excel sheet in the given format. Each image in the “images” directory would need to be read one by one and evaluated to extract and save the information present on it. The task also described a more challenging assignment of using a PDF file instead of evaluating individual images. 

## Installation

The codebase can be installed on any system, but was tested on windows and linux only. The steps to install the codebase is as follows:

- Verify that the correct version of Python is installed. The codebase was tested on Python 3.9. Clone the Github repository, and change directory to the repository.

```bash
git clone https://github.com/z404/DKNSB_Task
```

 - Install Tesseract-OCR. This step varies for each operating system. Specific installation instructions can be found [here](https://tesseract-ocr.github.io/tessdoc/Installation.html). To install on Debian based systems, the following command is used:
```bash
sudo apt-get install tesseract-ocr -y
```

 - Install required python libraries specified in requirements.txt using pip.
```bash
python3 -m pip install -r requirements.txt
```

- Start the backend FastAPI server using the given command
```bash
python3 -m uvicorn api:app
```

- In a separate console, start the frontend Streamlit app using the given command
```bash
python3 -m streamlit run frontend_app.py
```

## Usage

The codebase can be used in two ways. The first way is to use the frontend Streamlit app to upload the images and PDF file. The second way is to use the backend FastAPI server to upload the images and PDF file. To test the running of the codebase, the `main.py` file can be used.

Once the code is successfully run, the results are saved in `results.csv` file in the root directory. The results are also displayed on the frontend Streamlit app.