import logging
import numpy as np
import os
import os.path as op
import shutil
from misc import mkdir
from tsv_file import TSVFile
from tsv_file import generate_lineidx_file

#generate_lineidx_file('./metacoco/img.tsv', './metacoco/img.lineidx')
# generate_lineidx_file('./metacoco/hw.tsv', './metacoco/hw.lineidx')

# generate_lineidx_file('oscar_4343_base_0/pred.ordered_inserted_result_same_0_feature.test.beam5.max20.odlabels.tsv', 'oscar_4343_base_0/pred.ordered_inserted_result_same_0_feature.test.beam5.max20.odlabels.lineidx' )
# generate_lineidx_file('oscar_4343_base_bar1/pred.ordered_inserted_result_same_bar1_feature.test.beam5.max20.odlabels.tsv', 'oscar_4343_base_bar1/pred.ordered_inserted_result_same_bar1_feature.test.beam5.max20.odlabels.lineidx' )
# generate_lineidx_file('oscar_4343_base_bar2/pred.ordered_inserted_result_same_bar2_feature.test.beam5.max20.odlabels.tsv', 'oscar_4343_base_bar2/pred.ordered_inserted_result_same_bar2_feature.test.beam5.max20.odlabels.lineidx' )
# generate_lineidx_file('oscar_4343_base_bar3/pred.ordered_inserted_result_same_bar3_feature.test.beam5.max20.odlabels.tsv', 'oscar_4343_base_bar3/pred.ordered_inserted_result_same_bar3_feature.test.beam5.max20.odlabels.lineidx' )

# generate_lineidx_file( 'ori_oscar_large_5000/pred.coco_caption.test.beam5.max20.odlabels.tsv' , 'ori_oscar_large_5000/pred.coco_caption.test.beam5.max20.odlabels.lineidx' )
# generate_lineidx_file('ori_vinvl_base_5000/pred.coco_caption.test.beam5.max20.odlabels.tsv', 'ori_vinvl_base_5000/pred.coco_caption.test.beam5.max20.odlabels.lineidx')

# generate_lineidx_file( 'ori_vinvl_large_5000/pred.coco_caption.test.beam5.max20.odlabels.tsv', 'ori_vinvl_large_5000/pred.coco_caption.test.beam5.max20.odlabels.lineidx' )

# generate_lineidx_file( 'ratio_0/pred.ordered_inserted_result_same0.tsv', 'ratio_0/pred.ordered_inserted_result_same0.lineidx' )
# generate_lineidx_file( 'ratio_1/pred.ordered_inserted_result_same_bar1.tsv', 'ratio_1/pred.ordered_inserted_result_same_bar1.lineidx' )
# generate_lineidx_file( 'ratio_2/pred.ordered_inserted_result_same_bar2.tsv', 'ratio_2/pred.ordered_inserted_result_same_bar2.lineidx' )
# generate_lineidx_file( 'ratio_3/pred.ordered_inserted_result_same_bar3.tsv', 'ratio_3/pred.ordered_inserted_result_same_bar3.lineidx')
# generate_lineidx_file( 'ori_azure_cap/pred.val_test_5000.tsv', 'ori_azure_cap/pred.val_test_5000.lineidx' )
# generate_lineidx_file( 'ratio_0/pred.ordered_inserted_result_same0.tsv', 'ratio_0/pred.ordered_inserted_result_same0.lineidx' )
# generate_lineidx_file( 'ratio_bar1/pred.final_result_test_bar1.tsv', 'ratio_bar1/pred.final_result_test_bar1.lineidx' )
# generate_lineidx_file( 'azure_1000_0/pred.1000_test_final_result_test_0.tsv', 'azure_1000_0/pred.1000_test_final_result_test_0.lineidx' )
# generate_lineidx_file( 'azure_1000_bar1/pred.1000_test_final_result_test_bar1.tsv', 'azure_1000_bar1/pred.1000_test_final_result_test_bar1.lineidx')
# generate_lineidx_file( 'azure_1000_bar2/pred.1000_test_final_result_test_bar2.tsv', 'azure_1000_bar2/pred.1000_test_final_result_test_bar2.lineidx' )
# generate_lineidx_file( 'azure_1000_bar3/pred.1000_test_final_result_test_bar3.tsv', 'azure_1000_bar3/pred.1000_test_final_result_test_bar3.lineidx')


# generate_lineidx_file('ana_oscar_3280/pred.oscar_3280.test.beam5.max20.odlabels.tsv', 'ana_oscar_3280/pred.oscar_3280.test.beam5.max20.odlabels.lineidx')
# generate_lineidx_file('./oscar_3280/test.label.tsv', './oscar_3280/test.label.lineidx')

# tsv1 = TSVFile('./test.feature.tsv')
# tsv2 = TSVFile('./test.label.tsv')

generate_lineidx_file( 'azure_1000_0/pred.final_result_test_0.tsv', 'azure_1000_0/pred.final_result_test_0.lineidx' )
generate_lineidx_file( 'azure_1000_bar1/pred.final_result_test_bar1.tsv', 'azure_1000_bar1/pred.final_result_test_bar1.lineidx' )
generate_lineidx_file( 'azure_1000_bar2/pred.final_result_test_bar2.tsv', 'azure_1000_bar2/pred.final_result_test_bar2.lineidx' )
generate_lineidx_file( 'azure_1000_bar3/pred.final_result_test_bar3.tsv', 'azure_1000_bar3/pred.final_result_test_bar3.lineidx' )