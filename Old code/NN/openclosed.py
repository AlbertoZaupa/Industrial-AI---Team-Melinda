import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from celladataset import CellaDataset

# Programma che contiene il neural network per prevedere se la valvola è aperta oppure chiusa
# viene stabilito se la valvola é aperta oppure chiusa guardando il campo "Marcia Pompa Glicole"

TRAIN_FULL_PATH = os.path.join("..", "Valori", "better", "aprile_15min.csv")
TEST_FULL_PATH = os.path.join("..", "Valori", "better", "maggio_15min.csv")
MODEL_OUTPUT_FULL_PATH = os.path.join(
    "models", "most_recent", "openclosed", "look_behind_1", "model_openclosed{}.pth"
)
MODEL_ACCURACIES_FULL_PATH = os.path.join(
    "models", "most_recent", "openclosed", "look_behind_1", "accuracy.txt"
)
TEST_OUTPUT_FULL_PATH = os.path.join("tests", "test_openclosed2.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16

# Colonne da utilizzare per fare la previsione
# Il valore ad indice 0 è quello che voglio prevedere
open_closed_columns = [
    "Marcia Pompa Glicole",
    "Temperatura Cella",
    "Temperatura Mandata Glicole",
    "Temperatura Mele",
    "Temperatura Ritorno Glicole",
    "Temperatura Roccia 1",
]

# Utilizzo due trainset, uno che contiene solo valori pari a zero e uno che contiene valori diversi da zero
# Quello che faccio dopo è confrontare la loro lunghezza e con ConcatDataset cerco di avere un numero più o meno
# uguale di valori pari a zero e diversi da zero (altrimenti in fase di training se una delle due classi è molto più prevalente dell'altra
# il neural network impara a prevedere sempre i valori della classe maggioritaria)
trainset_zero = CellaDataset(
    TRAIN_FULL_PATH, numero_celle=[2, 6, 7, 8, 9, 10], colonne=open_closed_columns, filter="zero"
)
trainset_nonzero = CellaDataset(
    TRAIN_FULL_PATH, numero_celle=[2, 6, 7, 8, 9, 10], colonne=open_closed_columns, filter="nonzero"
)

print("Trainset zero length:", len(trainset_zero))
print("Trainset nonzero length:", len(trainset_nonzero))

# Come spiegato sopra concateno i due training set, imposto la batch size e metto lo shuffle
trainloader = DataLoader(
    ConcatDataset(
        [trainset_zero] + [trainset_nonzero] * 6,
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Genero il testset
testset = CellaDataset(TEST_FULL_PATH, [15, 18], open_closed_columns)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


def sigmoid_k(x, k):
    return 1.0 / (1.0 + torch.exp(-k * x))


# Misura l'accuracy del modello su un dataset (generalmente di test)
def measure_accuracy(model, dataloader, device="cpu"):
    right_answers = 0
    for batch in dataloader:
        inputs = batch["input"].to(device)
        labels = batch["label"].squeeze(1)
        outputs = model(inputs).squeeze(1)
        right_answers += torch.sum(torch.sum(torch.round(outputs) == labels))
    return right_answers / (len(dataloader) * batch_size)


# Classe del neural network
class OpenCloseNet(nn.Module):
    def __init__(self):
        super(OpenCloseNet, self).__init__()
        self.conv_layer_stack = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(2, 1)),
            nn.Sigmoid(),
            nn.Conv2d(1, 1, kernel_size=(2, 1)),
            nn.Sigmoid(),
        )
        self.fully_connected_layer_stack = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )        

    def forward(self, x: torch.Tensor):
        x = self.conv_layer_stack(x)
        x = self.fully_connected_layer_stack(x)
        x = sigmoid_k(x, 5)
        return x

# questa variabile consente di far caricare un modello precedentemente generato
# per valutarne ulteriormente l'accuratezza
load_saved_net = False

net = OpenCloseNet()
net = net.to(device)

if load_saved_net:
    net.load_state_dict(torch.load(MODEL_OUTPUT_FULL_PATH.format(30)))
else:
    epochs = 30
    learning_rate = 1e-3
    loss_print_update = 200

    loss_fn = nn.MSELoss()
    momentum = 0.9
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Inizio del training
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(trainloader):
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

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

        curr_accuracy = measure_accuracy(net, testloader, loss_fn, device)
        print(f"Accuracy at epoch {epoch + 1} is {(curr_accuracy):6f}")
        torch.save(net.state_dict(), MODEL_OUTPUT_FULL_PATH.format(epoch + 1))
        with open(MODEL_ACCURACIES_FULL_PATH, "a") as fp:
            fp.write(f"{epoch + 1} {curr_accuracy}\n")

    print("Finished Training")

# Viene stampato un csv con le predizioni del training set
testset.to_csv(TEST_OUTPUT_FULL_PATH, net, device=device, round_prediction=True)
print("Done!")
