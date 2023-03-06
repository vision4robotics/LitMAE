import torch
import torch.nn.functional as F

from .LitMAE.model import model_build as LitMAE
ENHANCERS = {
          'LitMAE': LitMAE,
         }

class Enhancer():
    def __init__(self, args):
        super(Enhancer, self).__init__()
        self.args = args
        if  args.enhancername.split('-')[0]=='LitMAE':
            self.model1 = LitMAE()
            self.model1.load_state_dict(torch.load(args.e_weights))
            self.model = self.model1.enhancer.cuda().eval()        
    def enhance(self, img):

        input_ = torch.div(img, 255.)
        if self.args.enhancername.split('-')[0]=='LitMAE':
            enhanced,r, n, x_masked, x_sp = self.model(input_)
        else:
            enhanced = self.model(input_)

        enhanced = torch.clamp(enhanced, 0, 1)

        return torch.mul(enhanced, 255.)


def build_enhancer(args):
    return Enhancer(args)

