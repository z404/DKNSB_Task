import os
import time

import cv2
import numpy as np
import pytesseract
from fastapi.responses import FileResponse
from pdf2image import convert_from_path


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
        dict_val["Country"] = lines[0]
        device_mode = False
        for i in lines[1:]:
            if "REF" in i: device_mode = False
            if len(i.rstrip()) == 0:
                continue
            elif ":" in i:
                key, value = i.split(":")
                dict_val[key] = value.strip()
            elif "REF" in i:
                key, value = i.split(" ")
                dict_val[key] = value.strip()
            else:
                if device_mode and "REF" not in i and i.lower() != "ce":
                    dict_val["Device Name"] += " "+i
            if "Device Name" in i: device_mode = True

        return dict_val

    def search_for_templates(self, image: np.ndarray, template_dir: str = "./symbols") -> str:
        """Searches for templates in an image and returns the result as a string"""
        list_of_images = []
        img1 = cv2.resize(image, (image.shape[1], image.shape[1]))
        height, width = 175, 210
        template_rows = [800, 800+height]
        template_cols = [150, 150+width, 150+2*width, 150+3*width]

        template_string = ""

        # check for ce
        if np.mean(img1[220:400, 1050:-125]) < 245:
            template_string += "4"

        # read symols directory
        symbols = []
        for i in range(1, 10):
            symbols.append(cv2.imread('./symbols/{}.png'.format(i))[:,:,:3])

        # fix 1st template
        symbols[0] = cv2.resize(symbols[0], (width - 30, height - 30))

        for row_num, row_val in enumerate(template_rows):
            for col_num, col_val in enumerate(template_cols):
                template = img1[row_val - 75:row_val+height + 75, col_val - 75:col_val+width + 75]
                template_crop = template[75:-75, 75:-75]
                if np.mean(template_crop) > 245:
                    continue
                
                max_temp_score, max_temp = 0, 0
                for symbol_num, symbol in enumerate(symbols):
                    result = cv2.matchTemplate(template, symbol, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    if max_val > max_temp_score:
                        max_temp_score = max_val
                        max_temp = symbol_num
                template_string += str(max_temp + 1)
        
        return template_string
        

    def write_to_file(self, information: dict, number: str) -> None:
        """Writes the result to a file"""
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
        
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], (1436, 761))
        return images

    def process_image(self, image: np.ndarray) -> dict:
        """Processes an image and returns the result as a dictionary"""
        # perform OCR
        text = self.perform_OCR_on_image(image)
        # process OCR text
        information = self.process_OCR_text(text)
        # search for templates
        number = self.search_for_templates(image)
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
        number = self.search_for_templates(image)
        self.write_to_file(information, number)
        timeend = time.time()
        print(f"Time taken: {timeend - timestart}")

    def all_images_test(self):
        for i in os.listdir("./images"):
            if ".jpg" not in i:
                continue
            self.single_img_test(f"./images/{i}")

    def pdf_test(self):
        images = self.read_pdf(f"./Label.pdf")
        for image in images:
            text = self.perform_OCR_on_image(image)
            information = self.process_OCR_text(text)
            number = self.search_for_templates(image)
            self.write_to_file(information, number)

if __name__ == "__main__":
    ImageProcessor().pdf_test()