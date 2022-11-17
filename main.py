import os
import time

import cv2
import numpy as np
import pytesseract
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import match_template
from skimage.io import imread
from pdf2image import convert_from_path, convert_from_bytes
from fastapi.responses import FileResponse

class ImageProcessor:
    def read_image(self, filename: str) -> np.ndarray:
        """Reads an image from a file and returns it as a numpy array"""
        return cv2.imread(filename)

    def perform_OCR_on_image(self, image: np.ndarray) -> str:
        """Performs OCR on an image and returns the result as a string"""
        # convert to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # thresholding
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # create rectangular kernel
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # apply dilation on the threshold image
        dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)

        # find contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # create a copy of original image
        im2 = image.copy()

        # draw contours on the copy of original image
        text = cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)

        # sort contours based on their area keeping minimum required area as '40' (anything smaller than this will not be considered)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        # generate ocr for each contour
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Drawing a rectangle on copied image
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]

            # Apply OCR on the cropped image
            text = pytesseract.image_to_string(cropped, config='--psm 11')

        return text

    def process_OCR_text(self, text: str) -> dict:
        """Processes the OCR text and returns the result as a string"""
        # split on newlines
        dict_val = {}
        lines = text.splitlines()
        for i in lines:
            if ":" in i:
                key, value = i.split(":")
                dict_val[key] = value.strip()
            elif "REF" in i:
                key, value = i.split(" ")
                dict_val[key] = value.strip()
        
        return dict_val

    def search_for_templates(self, image: np.ndarray, template_dir: str = "./symbols") -> str:
        """Searches for templates in an image and returns the result as a string"""
        list_of_images = []
        image = cv2.resize(image, (image.shape[1], image.shape[1]))
        image_gray = rgb2gray(image)

        for template_n in os.listdir(template_dir):
            if ".png" not in template_n:
                continue
            template = imread(f"{template_dir}/{template_n}")[:,:,:3]
            
            template_gray = rgb2gray(template)

            result = match_template(image_gray, template_gray)

            # check if the template is found
            if (np.max(result) > 0.48) or \
            (np.max(result) > 0.35 and template_n == "9.png") or \
            (np.max(result) > 0.38 and template_n == "7.png"):
                ij = np.unravel_index(np.argmax(result), result.shape)
                x, y = ij[::-1]
                col = 0
                if y < 700:
                    col = 1
                elif y < 900:
                    col = 2
                else:
                    col = 3

                list_of_images.append((x, col, template.shape[1], template.shape[0], template_n[0]))
        
        # plot the image
        for i in list_of_images:
            cv2.rectangle(image, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0, 255, 0), 2)


        return list_of_images

    def arrange_number(self, list_of_images: list) -> str:
        """Arranges the number and returns the result as a string"""
        arr = [[], [], [], []]
        for i in list_of_images:
            arr[i[1]].append(i)
        # sort each column by x
        for i in arr:
            i.sort(key=lambda x: x[0])
        
        # get the number
        number = ""
        for i in arr:
            for j in i:
                number += j[4]
        
        return number

    def write_to_file(self, information: dict, number: str) -> None:
        """Writes the result to a file"""
        self.clear_files()
        # check if result.csv exists
        if not os.path.exists("./result.csv"):
            with open("./result.csv", "w") as f:
                f.write(",".join(information.keys()) + ",Symbols\n")

        # write to file
        with open("./result.csv", "a") as f:
            f.write(",".join(information.values()) + "," + number + "\n")

        
    def clear_files(self) -> None:
        """Clears the result file"""
        try:
            os.remove("./result.csv")
        except:
            pass

        # remove all files from uploads
        for i in os.listdir("./uploads"):
            os.remove(f"./uploads/{i}")

    def read_pdf(self, filename: str) -> list:
        """Reads a pdf and returns the images as a list of numpy arrays"""
        images = []
        pages = convert_from_path(filename, 500)
        for page in pages:
            images.append(np.array(page))
        return images

    def process_image(self, image: np.ndarray) -> dict:
        """Processes an image and returns the result as a dictionary"""
        # perform OCR
        text = self.perform_OCR_on_image(image)
        # process OCR text
        information = self.process_OCR_text(text)
        # search for templates
        list_of_images = self.search_for_templates(image)
        # arrange number
        number = self.arrange_number(list_of_images)
        # add number to information
        information["Symbols"] = number

        return information
    
    def process_pdf(self, file: str) -> None:
        """Processes a pdf and writes the result to a file"""
        return_array = []
        # read the pdf
        images = self.read_pdf(file)
        # process each image
        for image in images:
            information = self.process_image(image)
            return_array.append(information)
        
        return return_array

    def create_csv(self, filetype: str, filepath: str = None, image: np.ndarray = None) -> None:
        """Creates a csv file from a pdf"""
        if filetype == "pdf":
            self.process_pdf(filepath)
        elif filetype == "image":
            information = self.process_image(image)
        
        self.write_to_file({i:j for i, j in information.items() if i != "Symbols"}, information["Symbols"])

        return FileResponse("./result.csv")

    ##############################
    #          Testing           #
    ##############################

    ## single image test
    def single_img_test(self, filename: str):
        timestart = time.time()
        image = self.read_image(filename)
        text = self.perform_OCR_on_image(image)
        information = self.process_OCR_text(text)
        list_of_images = self.search_for_templates(image)
        number = self.arrange_number(list_of_images)
        self.write_to_file(information, number)
        timeend = time.time()
        print(f"Time taken: {timeend - timestart}")

    def all_images_test(self):
        self.clear_file()
        for i in os.listdir("./images"):
            if ".jpg" not in i:
                continue
            self.single_img_test(f"./images/{i}")

    def pdf_test(self):
        self.clear_file()
        images = self.read_pdf(f"./Label.pdf")
        for image in images:
            text = self.perform_OCR_on_image(image)
            information = self.process_OCR_text(text)
            list_of_images = self.search_for_templates(image)
            number = self.arrange_number(list_of_images)
            self.write_to_file(information, number)

if __name__ == "__main__":
    ImageProcessor().pdf_test()