from templates import *
from dataset import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch

device = 'cuda:2'
conf = video_64_autoenc()
print(conf.name)
model = LitModel(conf)

state = torch.load(f'checkpoints/video_autoenc/last.ckpt', map_location=device)
#state = torch.load(f'checkpoints/video_autoenc/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=False)
model.model.eval()
model.model.to(device)

data = Video_Dataset(path="/home/eprakash/shanghaitech/testing/anomalies_list.txt", image_size=conf.img_size)
batch = data[11631]['img'][None]
frame_batch = data[11631]['frame_batch'][None]

print(batch.shape, frame_batch.shape)

print("Encoding...")
cond = model.encode(frame_batch.to(device))
print(cond.shape)
seed = np.random.randint(0, 1000000)
torch.manual_seed(seed)
#cond = torch.randn(1, 256, device=device)
print(cond.shape)
xT = model.encode_stochastic(x=frame_batch.to(device), cond=cond, T=250)
#with open('dummy_mnist_ex_4_encoding.txt', 'wb') as f:
#        np.save(f, xT.clone().cpu().detach().numpy())
#xT = torch.from_numpy(np.load("avg_st_train_encoding.txt")[None, :]).to(device)
#xT = torch.randn(1, 3, 9, conf.img_size, conf.img_size, device=device)
#xT = (xT_1 + xT_2)/2
print(xT.shape)

print("Decoding...")
cond = {'cond': cond}
pred = model.render(noise=xT, cond=cond, T=20)
ori = (batch + 1) / 2

print("Plotting...")
F = 9

fig, ax = plt.subplots(1, F, figsize=(F * 5, 5))
frame_batch = frame_batch.permute(0, 2, 1, 3, 4)
for i in range(F):
    img = frame_batch[0][i]
    ax[i].imshow(img.permute(1, 2, 0).cpu())
    #img_name = "ubnormal_normal_batch_ori/img_" + str(i) + ".png"
    #save_image(img.cpu(), img_name)
plt.savefig("shanghaitech_anomaly_ori_best.png")

fig, ax = plt.subplots(1, F, figsize=(F * 5, 5))
pred = pred.permute(0, 2, 1, 3, 4)
for i in range(F):
    img = pred[0][i]
    ax[i].imshow(img.permute(1, 2, 0).cpu())
    img_name = "st_anomaly_batch_gen_best/img_" + str(i) + ".png"
    save_image(img.cpu(), img_name)
plt.savefig("shanghaitech_anomaly_gen_best.png")

print("DONE!")
