from typing import Tuple, List
import pandas as pd
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import matplotlib as mpl
import seaborn as sns
from decimal import *
import os
import math


def read_annotations(path_to_labelTxt: str) -> Tuple[pd.DataFrame, List]:
    """
    Read the annotation files(txt) and create a data frame and a list that include the frames' names.

    Parameters
    ----------
    path_to_labelTxt : str
        The files location of the annotations.

    Returns
    -------
    dataframe
        A dataframe of the annotations includes 4 points of the bounding box of the annotation, category_id, and Frame name.
    list
        A list that includes the frames' names.
    """
    list_annotations_files = []
    list_frame_names = []
    columns_names = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'category_id']

    # iterate over files in the directory
    labelTxt_files = os.listdir(path_to_labelTxt)
    for filename in labelTxt_files:
        df = pd.read_csv(os.path.join(path_to_labelTxt, filename), sep=" ", header=None, names=columns_names)
        frame_name = filename.split(".")[0]
        df["Frame"] = frame_name
        list_annotations_files.append(df)
        list_frame_names.append(frame_name)

    ann_train = pd.concat(list_annotations_files)
    ann_train.reset_index(drop=True, inplace=True)

    return ann_train, list_frame_names


def bb_out_of_frame(annotations: pd.DataFrame, start_range_frame: int = 0, stop_range_frame: int = 1280):
    """
    Count the annotations (bounding boxes) that cross the frame size.

    Parameters
    ----------
    annotations : dataframe
        A dataframe of the annotations includes 4 points of the bounding box of the annotation, category_id, and Frame name.
    start_range_frame : int
        The lower limit range of square frame size.
    stop_range_frame = int
        The upper limit range of square frame size.

    Returns
    -------
    int
        The number of annotations (bounding boxes) that cross the frame size.
    """
    out_of_frame = annotations.loc[(annotations.x1<start_range_frame) | (annotations.x1>stop_range_frame) | (annotations.x2<start_range_frame) | (annotations.x2>stop_range_frame)| (annotations.x3<start_range_frame) | (annotations.x3>stop_range_frame) | (annotations.x4<start_range_frame) | (annotations.x4>stop_range_frame) |
                    (annotations.y1<start_range_frame) | (annotations.y1>stop_range_frame) | (annotations.y2<start_range_frame) | (annotations.y2>stop_range_frame)| (annotations.y3<start_range_frame) | (annotations.y3>stop_range_frame) | (annotations.y4<start_range_frame) | (annotations.y4>stop_range_frame)]
    return out_of_frame

def anno_vis_bar(annotations):
    """
    Visualization of total annotations per class

    Parameters
    ----------
    annotations : dataframe
        A dataframe of the annotations includes 4 points of the bounding box of the annotation, category_id, and Frame name..

    Returns
    -------
    -
    """
    ax = sns.countplot(data=annotations, y='category_id', order = annotations['category_id'].value_counts().index, palette = color_class)
    for container in ax.containers:
        ax.bar_label(container, labels=[f'{x:.0f}' for x in container.datavalues])
    ax.set_ylabel('category_id')
    ax.set_xlabel('annotation count')
    plt.title('Total annotations per class')
    plt.show()