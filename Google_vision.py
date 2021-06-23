import io
import os
import math
import pandas as pd

# Imports the Google Cloud client library
from google.cloud import vision

# Environment variable must be set before the function can be used
# Terminal command : set GOOGLE_APPLICATION_CREDENTIALS=<KEY_FILE_PATH>

def generate_annotations(input_file,output_file):
    """Procedure to generate a csv file with the output of googlevision API text detection

    Parameters
    ----------
    input_file : string
        Path of the picture file to be annotated
    output_file : string
        Path of the output csv file

    Output file content
    -------
    index : index of the box
    text : string
        the text of the given box
    box_center_x : float
        the x coordinate of the center of the given box
    box_center_y : float
        the y coordinate of the center of the given box
    box_width : int
        the width of the given box
    box_height : int
        the height of the given box
    """
    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(input_file, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Performs text detection on the image file
    response = client.text_detection(image=image)
    
    # Initialize a Dictionnary to store the results
    result = {'text':[],'box_center_x':[],'box_center_y':[],'box_width':[],'box_height':[]}
    
    #loop over the boxes
    for box in response.text_annotations:
        #Text content of the box
        text_content = box.description
        
        #Initialize the minimum and maximum coordinates of the box
        min_x = math.inf
        max_x=0
        min_y = math.inf
        max_y=0

        # Loop over the corners of the box to get the minimum and maximum coordinates of
        # the corners of the box
        for corner in box.bounding_poly.vertices:
            if corner.x < min_x:
                min_x = corner.x
            if corner.x > max_x:
                max_x = corner.x
            if corner.y < min_y:
                min_y = corner.y
            if corner.y > max_y:
                max_y = corner.y
        
        # Calculate the center, width and height of the box
        box_width = max_x - min_x
        box_height = max_y - min_y
        box_center_x = min_x + (box_width / 2)
        box_center_y = min_y + (box_height / 2)

        # Store the results in the Dictionnary
        result['text'].append(text_content)
        result['box_center_x'].append(box_center_x)
        result['box_center_y'].append(box_center_y)
        result['box_width'].append(box_width)
        result['box_height'].append(box_height)

    # Store the results in the output file
    result_df = pd.DataFrame(result)
    result_df.to_csv(output_file)

# To Test of the function, uncomment the following lines
#file_name = os.path.abspath('1193-receipt.jpg')
#generate_annotations(file_name,'output.csv')
