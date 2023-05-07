import torch
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import os
def compute_ca_loss(attn_maps_mid, attn_maps_up, bboxes, object_positions):
    loss = 0
    object_number = len(bboxes)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    for attn_map_integrated in attn_maps_mid:
        #This chunks the attention mid from [8,64,77] to [4,64,77]. We pick the second half. Why?
        attn_map = attn_map_integrated.chunk(2)[1]
        #Then we extract the dimensions. b= 4 i=64 j=77
        b, i, j = attn_map.shape
        #Not exactly sure why we take sqrt of dimension i. Probably beause I is the flattened size of the input. Like 8x8.
        H = W = int(math.sqrt(i))
        #Loop for the amount of objects (basically for how many bounding boxes we created)
        for obj_idx in range(object_number):
            obj_loss = 0 #per object "loss"
            #We create a mask of all zeros using the sqrt i dimension, in this case 8x8
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_box in bboxes[obj_idx]:
                #Extract the corners of the bounding boxes and set the mask matrix to 1 at locations inside the bounding box.
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1
            #Object_position in example case looks like: [[2,3],[10]]
            for obj_position in object_positions[obj_idx]: #(1) 2 3 (2) 10
                #Third dimension corresponds then to attention map of words. Picking the location of the word that has a bounding box means getting that specific attention for that word.
                #Originally attn_map[:,:,obj_position] has shape [4,64,1]
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W) #Reshape the specific attention map of that word into [4,8,8]
                '''
                obj_position is an integer which corresponds to the index in the query of that specific word. So here we assume that selecting the index will give us the attention
                map of each word in the query. But why is it length 77?
                '''
                #Multiply this specific attention map with the mask, merge the final two dimensions and sum along them. We divide by the specific attention map of this particular word.
                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                #Intuitively: We amplify attention values of this specific object within the bounding box (indicated by mask), to then guide to model to attend more to this area for this object?
                obj_loss += torch.mean((1 - activation_value) ** 2) #if for example an object has multiple words associated with it like "hello kitty", we sum the "losses" of both words.
            loss += (obj_loss/len(object_positions[obj_idx]))
    #Then the same stuff for the upsample blocks, not sure if this is something that is essential and how to transfer this to point-e
    for attn_map_integrated in attn_maps_up[0]:
        attn_map = attn_map_integrated.chunk(2)[1]
        #
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                
                # ca_map_obj = attn_map[:, :, object_positions[obj_position]].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(
                    dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss / len(object_positions[obj_idx]))
    loss = loss / (object_number * (len(attn_maps_up[0]) + len(attn_maps_mid))) #some average calculation across mid and up blocks.
    return loss

def Pharse2idx(prompt, phrases):
    #Example: "hello kitty; ball",
    phrases = [x.strip() for x in phrases.split(';')] #phrases = [hello kitty, ball]
    # "A hello kitty toy is playing with a purple ball."
    prompt_list = prompt.strip('.').split(' ') #prompt_list = ["A", "hello", "kitty", "toy", "is", "playing", "with", "a","purple", "ball"]
    object_positions = []
    for obj in phrases: #only hello kitty and ball
        obj_position = []
        for word in obj.split(' '): #word <- (1) hello , kitty (2) ball
            obj_first_index = prompt_list.index(word) + 1 #for "hello" obj_first_index = 2, for "kitty" obj_first_index = 3, for "ball" obj_first_index = 10
            obj_position.append(obj_first_index) 
        object_positions.append(obj_position) #[[2,3],[10]]

    return object_positions

def draw_box(pil_img, bboxes, phrases, save_path):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype('./FreeMono.ttf', 25)
    phrases = [x.strip() for x in phrases.split(';')]
    for obj_bboxes, phrase in zip(bboxes, phrases):
        for obj_bbox in obj_bboxes:
            x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
            draw.rectangle([int(x_0 * 512), int(y_0 * 512), int(x_1 * 512), int(y_1 * 512)], outline='red', width=5)
            draw.text((int(x_0 * 512) + 5, int(y_0 * 512) + 5), phrase, font=font, fill=(255, 0, 0))
    pil_img.save(save_path)



def setup_logger(save_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(os.path.join(save_path, f"{logger_name}.log"))
    file_handler.setLevel(logging.INFO)

    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger