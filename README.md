# Simple Motion Detection in Video Sequences
*A Digital Image and Processing Project* (2019/2020)

This contains the implementation of the DIPA group project on simple motion detection in video sequences. There are
two implementations to be found -- the background subtraction model, and the optical flow-based model.

**main_background_subtraction.py** and **main_optical_flow.py** contain the functions used. Their prototypes are similar. 
The parameters they both share are:

* foldPath -- if the video file is a set of images (frames), this is the path to the folder. if a video file, this is the path to the video file itself
* extension -- the file extension of image frames (ignored if it's a video file)
* vidFile -- flag to determine whether or not we're dealing with a video file or a set of images
* frameDigits -- used only for image frames; the number of digits expected for the image files to have (meant to be used for zero-padded images)
* extraPrefix -- used only for image frames; if there is a prefix set in front of the zero-padded images 
* nopad -- used only for image frames; set to True if the image files have no zero-padding
* minSize -- the minimum area of a foreground object 
* *(OPTICAL FLOW IMPLEMENTATION ONLY)* tresh_val -- magnitude threshold of optical flow vectors
* resiz_am -- parameter to resize the image width (the height keeps the aspect ratio)

The methods also include optional methods to help analyze the ground truth However, usable only if the input is image frames! 
It is assumed that the ground truth follows the same numbering conventions as the dataset images (IE, whether or not it uses zero-padding and the overall number of digits used if it does)
* withTest -- flag to determine whether or not the test mode should be on
* gt_foldPath -- folder path to the set of images containing the ground truth for every frame
* gt_extension -- file extention of the ground truth images
* gt_extraPrefix -- prefix in front of the numbering

If the *withTest* flag is not set, the default system behavior is to simply display the foreground and the frames (the latter having bounding boxes if movement is found).
If it is set, then a third window will be visible, one containing the bounding box of the ground truth. **Bear in mind that performance will take a major hit while running the methods in test mode!**

The **testing.py** is a test suite designed to run on the LASIESTA dataset, which can be found here: http://www.gti.ssr.upm.es/data/LASIESTA

Unfortunately, the LASIESTA dataset needs to have its files downloaded individually. The videos you will require are:

        'I_BS_01',
        'I_BS_02',
        'I_IL_01',
        'I_IL_02',
        'I_MB_02',
        'I_OC_01',
        'I_OC_02',
        'I_SI_01',
        'I_SI_02',
        'O_CL_01',
        'O_CL_02',
        'O_RA_01',
        'O_RA_02',
        'O_SN_01',
        'O_SN_02',
        'O_MC_01',
        'O_MC_02',
        'O_SU_01',
        'O_SU_02'

To run the tests:
* Create a new folder in the same folder as **testing.py** named *LASIESTA*
* Extract all of the downloaded files into that folder
* Run **testing.py**
* Accuracy and relative accuracy will be displayed in the output console

The **results.txt** contains the results of our own experimentation.

**resultparse.py** is used to simply parse the results in the **results.txt** file and calculate the average accuracies and relative accuracies.