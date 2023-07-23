import torch
import torch.nn as nn



class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10

class PSNR_RGB(nn.Module):
    def __init__(self, max_val):
        super(PSNR_RGB, self).__init__()

    def __call__(self, a, b):
        mse = torch.mean((a.float()-b.float())**2)

        if mse == 0:
            return torch.tensor(0)

        psnr = 10*torch.log10(255*255 / mse)

        return psnr

class PSNR_YCbcr(nn.Module):
    def __init__(self):
        super(PSNR_YCbcr, self).__init__()

    def __call__(self, a, b):
        a = a.float()[0]
        b = b.float()[0]
        Y_a = 0.256789*a[...,0] + 0.504129*a[...,1] + 0.097906*a[...,2] + 16
        Y_b = 0.256789*b[...,0] + 0.504129*b[...,1] + 0.097906*b[...,2] + 16

        mse = torch.mean((Y_a-Y_b)**2)
        if mse == 0:
            return torch.tensor(0)

        psnr = 10*torch.log10(255*255/mse)

        return psnr