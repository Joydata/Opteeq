import io
import os
import math
import pandas as pd

# Imports the Google Cloud client library
from google.cloud import vision

# Environment variable must be set before the function can be used
# Terminal command : set GOOGLE_APPLICATION_CREDENTIALS=<KEY_FILE_PATH>

def generate_annotations(input_file, output_file):
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
    box_x : int
        the x coordinate of the top left corner of the given box
    box_y : int
        the y coordinate of the top left corner of the given box
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
    result = {'text':[],'box_x':[],'box_y':[],'box_width':[],'box_height':[]}
    
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
        
        # Calculate the coordinates, width and height of the box
        box_width = max_x - min_x
        box_height = max_y - min_y
        box_x = min_x
        box_y = min_y

        # Store the results in the Dictionnary
        result['text'].append(text_content)
        result['box_x'].append(box_x)
        result['box_y'].append(box_y)
        result['box_width'].append(box_width)
        result['box_height'].append(box_height)

    # Store the results in the output file
    result_df = pd.DataFrame(result)
    result_df.to_csv(output_file)

def convert_annotations(input_image, input_annotations, output_file):
    """Procedure to convert an annotation file for further import into VIA

    Parameters
    ----------
    input_image : string
        Path of the image file
    input_annotations : string
        Path of the csv file with annotations generated with generate_annotations function
    output_file : string
        Path of the output csv file

    Output file
    -------
    csv file with the right format to import in VIA
    """
    # Load the annotations in a DataFrame
    annotations = pd.read_csv(input_annotations)

    # Initialize a Dictionnary to store the results
    result={'filename':[],'file_size':[],'file_attributes':[],'region_count':[],'region_id':[],'region_shape_attributes':[],'region_attributes':[]}
    
    # Prepare the file information    
    file_size = os.path.getsize(input_image)
    file_attributes = '{"caption":"","public_domain":"no","image_url":""}'
    
    # Prepare the count of annotations 
    region_count = len(annotations)

    # Loop over annotations
    for idx in annotations.index:
        # Get coordinates of the box
        X = int(annotations.iloc[idx]['box_x'])
        Y = int(annotations.iloc[idx]['box_y'])
        WIDTH = annotations.iloc[idx]['box_width']
        HEIGHT = annotations.iloc[idx]['box_height']
        
        # Get the text inside the box. New-line characters are replaced by space characters
        TEXT = annotations.iloc[idx]['text'].replace('\n',' ')
        
        # Prepare the region_shape attributes and region attributes 
        region_shape_attributes = f'{{"name":"rect","x":{X},"y":{Y},"width":{WIDTH},"height":{HEIGHT}}}'
        region_attributes = f'{{"name":"{TEXT}","type":"unknown","image_quality":{{"good":true,"frontal":true,"good_illumination":true}}}}'
        
        # Store all information in the dictionnary
        result['filename'].append(input_image)
        result['file_size'].append(file_size)
        result['file_attributes'].append(file_attributes)
        result['region_count'].append(region_count)
        result['region_id'].append(idx)
        result['region_shape_attributes'].append(region_shape_attributes)
        result['region_attributes'].append(region_attributes)
    
    # Store the results in the output file
    result_df = pd.DataFrame(result)
    result_df.to_csv(output_file, header=True, index=False)


# To Test of the functions, uncomment the following lines
#file_name = '1193-receipt.jpg'
#generate_annotations(file_name, 'annotations.csv')
#convert_annotations(file_name, 'annotations.csv', 'via_csv.csv')
