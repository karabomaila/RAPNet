from RAP import RAP

def train():
    model = RAP()
    model.cuda()
    model.train()
    