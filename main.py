from pytorch_lightning import Trainer

from AEN import AEN

model = AEN(k=4, n_channels=7)

trainer = Trainer(max_epochs=50, early_stop_callback=False)

trainer.fit(model)

trainer.test(model)
