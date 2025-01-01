from DriveSceneGen.scripts.train import TrainingConfig, training_mine

if __name__ == "__main__":
         
    training_mine(NUM_EPOCHS=3, INFER_STEPS=1000)