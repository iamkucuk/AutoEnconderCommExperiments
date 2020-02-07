from pytorch_lightning import Trainer

from AEN import AEN

model = AEN()

trainer = Trainer(max_epochs=10, early_stop_callback=False)

trainer.fit(model)

trainer.test(model)
