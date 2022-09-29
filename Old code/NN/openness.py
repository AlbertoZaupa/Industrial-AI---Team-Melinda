import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from celladataset import CellaDataset

# Programma che contiene il neural network per prevedere l'apertura della valvola

TRAIN_FULL_PATH = os.path.join("..", "Valori", "better", "all_nn.csv")
TEST_FULL_PATH = os.path.join("..", "Valori", "better", "all_nn.csv")
MODEL_OUTPUT_FULL_PATH = os.path.join(
    "models",
    "openness",
    f"look_behind_{CellaDataset.look_behind}",
    "model_openness_1.{}.pth",
)
MODEL_LOSSES_FULL_PATH = os.path.join(
    "models", "openness", f"look_behind_{CellaDataset.look_behind}", "accuracy.txt"
)
TEST_OUTPUT_FULL_PATH = os.path.join("tests", "openness4.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16

# Colonne da utilizzare per fare la previsione
# Il valore ad indice 0 è quello che voglio prevedere
openness_columns = [
    "Apertura Valvola Miscelatrice",
    "Temperatura Cella",
    "Temperatura Mandata Glicole",
    "Temperatura Mele",
    "Temperatura Roccia 2",
    "Temperatura Roccia 3",
    "Temperatura Roccia 4",
]

# Utilizzo due trainset, uno che contiene solo valori pari a zero e uno che contiene valori diversi da zero
# Quello che faccio dopo è confrontare la loro lunghezza e con ConcatDataset cerco di avere un numero più o meno
# uguale di valori pari a zero e diversi da zero (altrimenti in fase di training se una delle due classi è molto più prevalente dell'altra
# il neural network impara a prevedere sempre i valori della classe maggioritaria)
trainset_zero = CellaDataset(
    TRAIN_FULL_PATH,
    [2, 6, 7, 8, 9, 10, 15, 18],
    openness_columns,
    months=[3, 4, 5, 6],
    filter="zero",
)
trainset_nonzero = CellaDataset(
    TRAIN_FULL_PATH,
    [2, 6, 7, 8, 9, 10, 15, 18],
    openness_columns,
    months=[3, 4, 5, 6],
    filter="nonzero",
)

print("Trainset zero length:", len(trainset_zero))
print("Trainset nonzero length:", len(trainset_nonzero))

# Come spiegato sopra concateno i due training set, imposto la batch size e metto lo shuffle
trainloader = DataLoader(
    ConcatDataset([trainset_zero] + [trainset_nonzero] * 7,),
    batch_size=batch_size,
    shuffle=True,
)

# Genero il testset
testset = CellaDataset(
    TEST_FULL_PATH, [2, 6, 7, 8, 9, 10, 15, 18], openness_columns, months=[1, 2]
)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


# Misura la loss del modello su un dataset (generalmente di test)
def measure_loss(model, dataloader, loss_fn, device="cpu"):
    with torch.no_grad():
        total_loss = 0.0
        for batch in dataloader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).item()
            total_loss += loss
        return total_loss / len(dataloader)

# Classe del neural network
class OpennessNet(nn.Module):
    def __init__(self):
        super(OpennessNet, self).__init__()
        self.conv_layer_stack = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(2, 1)),
            nn.Sigmoid(),
            nn.Conv2d(1, 1, kernel_size=(2, 1)),
            nn.Sigmoid(),
        )
        self.fully_connected_layer_stack = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(14, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_layer_stack(x)
        x = self.fully_connected_layer_stack(x)
        return x

# questa variabile consente di far caricare un modello precedentemente generato
# per valutarne ulteriormente l'accuratezza
load_saved_net = True

net = OpennessNet()
net = net.to(device)

if load_saved_net:
    net.load_state_dict(torch.load(MODEL_OUTPUT_FULL_PATH.format(7)))
else:
    epochs = 30
    learning_rate = 1e-3
    loss_print_update = 200
    # loss_print_update = 1

    # Ho provato diverse loss function tra cui RMSE e MSE,
    # in generale MSE sembra andare meglio la maggior parte delle volte
    # loss_fn = lambda output, label: torch.sqrt(torch.nn.MSELoss()(output, label))
    loss_fn = torch.nn.MSELoss()

    # Stessa cosa per gli optimizer, sembra andare meglio con SGD e momentum
    momentum = 0.9
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Inizio del training
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(trainloader):
            inputs: torch.Tensor = batch["input"].to(device)
            labels: torch.Tensor = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss: torch.Tensor = loss_fn(outputs, labels)
            running_loss += loss

            loss.backward()
            optimizer.step()

            if i % loss_print_update == (loss_print_update - 1):
                print(
                    f"[{epoch + 1}, {i + 1}] loss: {(running_loss.item() / loss_print_update):.6f}"
                )
                running_loss = 0.0

        curr_loss = measure_loss(net, testloader, loss_fn, device)
        print(f"Loss at epoch {epoch + 1} is {(curr_loss):6f}")
        torch.save(net.state_dict(), MODEL_OUTPUT_FULL_PATH.format(epoch + 1))
        with open(MODEL_LOSSES_FULL_PATH, "a") as fp:
            fp.write(f"{epoch + 1} {curr_loss}\n")

    print("Finished Training")

# Viene stampato un csv con le predizioni del training set
testset.to_csv(TEST_OUTPUT_FULL_PATH, net, device=device)
print("Done!")
