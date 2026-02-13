import torch.nn as nn  # Importerar PyTorchs modul för neurala nätverk

# Vi skapar en egen modellklass som ärver från nn.Module
# Alla träningsbara modeller i PyTorch bygger på nn.Module
class SimpleClassifier(nn.Module):

    def __init__(self):
        # Anropar konstruktor i nn.Module (viktigt för att PyTorch ska funka korrekt)
        super().__init__()

        # Här definierar vi själva nätverkets arkitektur
        # nn.Sequential betyder: lager körs i ordning
        self.model = nn.Sequential(
            # Första lagret:
            # Tar en input-vektor med 3072 värden (3*32*32 pixlar)
            # och mappar till 128 neuroner
            nn.Linear(3072, 128),

            # ReLU = aktiveringsfunktion
            # Lägger in icke-linearitet så modellen kan lära komplexa mönster
            nn.ReLU(),

            # Output-lagret:
            # Mappar från 128 neuroner till 10 klasser (CIFAR-10 har 10 kategorier)
            nn.Linear(128, 10)
        )

    # forward beskriver hur data flödar genom nätverket
    # Den körs automatiskt när vi gör: model(x)
    def forward(self, x):

        # Bilder kommer in som: [batch_size, 3, 32, 32]
        # Vi plattar ut varje bild till en vektor: [batch_size, 3072]
        x = x.view(x.size(0), -1)  # flatten image

        # Skickar datan genom vårt nätverk
        return self.model(x)
