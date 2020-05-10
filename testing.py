import numpy as np
import main_background_subtraction
import main_optical_flow

mainfolds = [
    './LASIESTA'
]

sub_folds = [
    [
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
    ]
]

to_test = {
    'BACKGROUND SUBTRACTION': main_background_subtraction.motion_tracking,
    'OPTICAL FLOW': main_optical_flow.motion_tracking,

}

min_sizes = {
    'BACKGROUND SUBTRACTION': 2500,
    'OPTICAL FLOW': 2000,
}


for t in to_test:
    print(t)
    for i in range(len(mainfolds)):
        m = mainfolds[i]
        for s in sub_folds[i]:
            foldpath = m + '/' + s
            scores, scores_r = to_test[t](foldpath + '/', extension='.bmp', vidFile=False, extraPrefix=s + '-', minSize=min_sizes[t], nopad=True,
                                             withTest=True, gt_foldPath=foldpath + '-GT/', gt_extension='.png', gt_extraPrefix=s + '-GT_')
            print('ACCURACY: ', s, ':', np.sum(scores) / scores.shape[0], "RELEVANT ACCURACY:", np.sum(scores_r) / scores_r.shape[0])

    print('====================\n')