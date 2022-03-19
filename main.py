from imageai.Detection import ObjectDetection
import glob
import re
from dotenv import dotenv_values

config = dotenv_values(".env")
detector = ObjectDetection()
model_path = "./model/" . config["MODEL"]

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
jpg = glob.glob("./input/*.jpg")
files = jpg + glob.glob("./input/*.jpeg")

for file in files:
  nameFile = re.sub(r"^.{8}", "", file, 0, re.MULTILINE)
  output_path = "./output/"+ nameFile
  detection = detector.detectObjectsFromImage(input_image=file, output_image_path=output_path)
  print(file + " reconheceu: ")
  for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])
