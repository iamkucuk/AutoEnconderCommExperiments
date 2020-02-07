from pytorch_lightning import Trainer

from AEN import AEN

model = AEN(n_channels=7)

trainer = Trainer(max_epochs=10, early_stop_callback=False)

trainer.fit(model)

trainer.test(model)
