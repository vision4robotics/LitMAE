from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals



from snot.pipelines.siamapn_pipeline import SiamAPNPipeline
from snot.pipelines.siamapnpp_pipeline import SiamAPNppPipeline
from snot.pipelines.siamrpn_pipeline import SiamRPNppPipeline
from snot.pipelines.hift_pipeline import HiFTPipeline
from snot.pipelines.lpat_pipeline import LPATPipeline


TRACKERS =  {
          'SiamAPN': SiamAPNPipeline,
          'SiamAPN++': SiamAPNppPipeline,
          'SiamRPN++': SiamRPNppPipeline,
          'HiFT':HiFTPipeline,
          'LPAT':LPATPipeline,

          'MAESiamAPN': SiamAPNPipeline,
          'MAESiamAPN++': SiamAPNppPipeline,
          'MAESiamRPN++': SiamRPNppPipeline,
          'MAEHiFT':HiFTPipeline,
          'MAELPAT':LPATPipeline,
          
          

          }

def build_pipeline(args, enhancer):
    return TRACKERS[args.trackername.split('_')[0]](args, enhancer)

