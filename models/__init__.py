def create_model(opt):
    model = opt['model']##this para is came from the .json file
    
    #the model in jason, decided which modl import
    #so if you add a new model, this .py must be modified
    if model == 'sr':###this is the SR model
        from .SR_model import SRModel as M#take sr as an example
############################################################################################3
    elif model == 'sr_sub':###this is the SRGAN
        from .SR_subnet_model import SRModel as M
#######################################################################################
    elif model == 'srgan':###this is the SRGAN
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':###this is the SFTGAN
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    print('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m#return the model
