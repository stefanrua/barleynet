def get_tiles(img, tile_w, tile_h):
    """
    A generator that yields tiles from a large image.
    
    Args:
        img (np.array): The image to tile.
        tile_w (int): The width of a tile.
        tile_h (int): The height of a tile.
        
    Yields:
        tuple: A tuple containing tile_coords ([x1, y1, x2, y2]), 
               the tile_image (np.array), and the tile_number.
    """
    img_h, img_w, _ = img.shape
    offset_x = 0
    offset_y = 0
    tile_n = 0
    
    while offset_y < img_h:
        max_y = min(offset_y + tile_h, img_h)
        while offset_x < img_w:
            max_x = min(offset_x + tile_w, img_w)
            
            tile_coords = [offset_x, offset_y, max_x, max_y]
            tile_img = img[offset_y:max_y, offset_x:max_x, :]
            
            yield tile_coords, tile_img, tile_n
            
            tile_n += 1
            offset_x += tile_w
        offset_x = 0
        offset_y += tile_h

def xywh_to_x1y1x2y2(bbox):
    """Converts bbox from [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def x1y1x2y2_to_xywh(bbox):
    """Converts bbox from [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

def bbox_in_tile(tile_coords, bbox_coords):
    """
    Checks if a bounding box overlaps with a tile.
    
    Args:
        tile_coords (list): [x1, y1, x2, y2] for the tile.
        bbox_coords (list): [x1, y1, x2, y2] for the bounding box.
        
    Returns:
        bool: True if they overlap, False otherwise.
    """
    tile_x1, tile_y1, tile_x2, tile_y2 = tile_coords
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_coords

    # Check for intersection
    if (tile_x2 < bbox_x1 or tile_x1 > bbox_x2 or
        tile_y2 < bbox_y1 or tile_y1 > bbox_y2):
        return False
    return True

def get_overlapping_part(tile_coords, bbox_coords):
    """
    Calculates the overlapping part of a tile and a bounding box.
    
    Args:
        tile_coords (list): [x1, y1, x2, y2] for the tile.
        bbox_coords (list): [x1, y1, x2, y2] for the bounding box.
        
    Returns:
        list: The overlapping bounding box [x1, y1, x2, y2] or None.
    """
    xmin = max(tile_coords[0], bbox_coords[0])
    xmax = min(tile_coords[2], bbox_coords[2])
    ymin = max(tile_coords[1], bbox_coords[1])
    ymax = min(tile_coords[3], bbox_coords[3])
    
    if (xmax - xmin > 0) and (ymax - ymin > 0):
        return [xmin, ymin, xmax, ymax]
    return None 