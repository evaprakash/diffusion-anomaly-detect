from templates import *
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision import models
torch.cuda.empty_cache()

device = 'cuda:0'
conf = video_64_autoenc()
print(conf.name)
model = LitModel(conf)

state = torch.load(f'checkpoints/video_autoenc/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=False)
model.model.eval()
model.model.to(device)

#model.kl_model.eval()
#model.kl_model.to(device)
#for k in model.kl_model.parameters():
#    k.requires_grad = False

data = Video_Dataset(path="/home/eprakash/shanghaitech/testing/anomalies_list.txt", image_size=conf.img_size)

batch_size = 6
num_batches = int(len(data)/batch_size) + 1
print(num_batches)
avg_loss = []

for b in range(num_batches):
    
    print("Building batch...")
    batch_len = batch_size
    if (b == num_batches - 1):
        batch_len = len(data) - batch_size * b
    print("Batch len: " + str(batch_len))
    batch = torch.zeros(size=((batch_len, 3, conf.img_size, conf.img_size)))
    frame_batch = torch.zeros(size=((batch_len, 3, 9, conf.img_size, conf.img_size))) 
    for i in range(batch_len):
        batch[i] = data[batch_size * b + i]['img'][None]
        frame_batch[i] = data[batch_size * b + i]['frame_batch'][None]
        if (i % 10 == 0):
            print("Done " + str(i))
    
    print("Encoding...")
    cond = model.encode(frame_batch.to(device))
    #name = 'st_anomaly_semantic_128/'+ str(b) + '.txt'
    #with open(name, 'wb') as f:
    #    np.save(f, cond.clone().cpu().detach().numpy())
    #seed = np.random.randint(0, 1000000)
    #torch.manual_seed(seed)
    #xT = torch.from_numpy(np.load("avg_st_train_encoding_256_4_full.txt")[None, :]).to(device).repeat(batch_len, 1, 1, 1, 1)
    xT = model.encode_stochastic(frame_batch.to(device), cond, T=250)
    xT = torch.permute(xT, (2, 0, 1, 3, 4))
    xT = torch.mean(xT, dim=0)[None, :]
    xT = xT.repeat(9, 1, 1, 1, 1) 
    xT = torch.permute(xT, (1, 2, 0, 3, 4)).to(device)
    print(xT.shape)
    #name = 'st_anomaly_stochastic_256/'+ str(b) + '.txt'
    #with open(name, 'wb') as f:
    #    np.save(f, xT.clone().cpu().detach().numpy())
    #xT_2 = torch.from_numpy(np.load("dummy_mnist_train_encoding.txt")).to(device).repeat(batch_len, 1, 1, 1, 1)#.view(batch_len, 3, 9, conf.img_size, conf.img_size)
    #xT = (xT_1 + xT_2)/2
    #torch.randn(batch_len, 3, 9, conf.img_size, conf.img_size).to(device)#model.encode_stochastic(frame_batch.to(device), cond, T=250)
    
    print("Decoding...")
    cond = {'cond': cond}
    pred = model.render(xT, cond, T=20)
    ori = frame_batch#(batch + 1) / 2
    ori_tensor = ori.clone().cpu()
    pred_tensor = pred.clone().cpu()
    
    '''
    if (b == 0):
        print("Saving sample image...")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(ori_tensor[0].permute(1, 2, 0).cpu())
        ax[1].imshow(pred_tensor[0].permute(1, 2, 0).cpu())
        plt.savefig("anomaly_img_0.png")
    '''
    print("Calculating losses...")
    abs_diff = torch.abs(pred.to(device) - ori.to(device))
    abs_diff = torch.flatten(abs_diff, start_dim=1)
    top = torch.topk(abs_diff, 11059, dim=1)[0]
    mean_diff = torch.mean(top, dim=1)
    with open("st_anomaly_diffs_256_4_single_enc.log", "a") as fp:
        fp.write("Batch diffs: ")
        for i in range(len(mean_diff)):
            fp.write(str(mean_diff[i].item()) + "|")
        fp.write("\n")
    #print(mean_diff)
    diff = torch.square(pred.to(device) - ori.to(device))
    diff = diff.reshape(batch_len, 9, 3, conf.img_size, conf.img_size)
    diff_flat = torch.flatten(diff, start_dim=2)
    diff = torch.mean(diff_flat, dim=2)
    with open("st_anomaly_mse_256_4_single_enc.log", "a") as fp:
        fp.write("Batch scores: ")
        batch_mean = torch.mean(diff, dim=1)
        for i in range(len(batch_mean)):
            fp.write(str(batch_mean[i].item()) + "|")
        fp.write("\n")
    scores = 10 * torch.log10(torch.div(1, diff))
    min_scores = torch.min(scores, dim=1)[0]
    max_scores = torch.max(scores, dim=1)[0]
    score = torch.div(torch.subtract(scores[:, 4], min_scores), torch.subtract(max_scores, min_scores))
    with open("st_anomaly_scores_256_4_single_enc.log", "a") as fp:
        fp.write("Batch scores: ")
        for i in range(len(score)):
            fp.write(str(score[i].item()) + "|")
        fp.write("\n")
    '''
    with torch.no_grad():
        loss = model.kl_model.forward(pred.to(device), ori.to(device))
        print("Losses: " + str(loss))
        print("Batch loss: " + str(torch.mean(loss).item()))
        with open("anomaly_losses.log", "a") as fp:
            fp.write("Batch loss: " + str(torch.sum(loss).item()) + "\n")
        avg_loss.append(torch.mean(loss).item())
    '''
    print("Done batch " +  str(b))
#print("Average loss: " +  str(np.mean(avg_loss)))
print("DONE!")
