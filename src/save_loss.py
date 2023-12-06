import pandas as pd

def save_loss(
        validationloader,
        savepath,
        epoch,
        loss,
        F1_micro,
        F1_macro,
        val_loss=None,
        F1_val_micro=None,
        F1_val_macro=None,
        ):
    if validationloader is not None:
        if epoch == 0:
            table = pd.DataFrame(
                {
                    'epoch': [epoch],
                    'loss': [loss], 
                    'val_loss': [val_loss], 
                    'F1_micro': [F1_micro], 
                    'F1_macro': [F1_macro],
                    'F1_val_micro': [F1_val_micro],
                    'F1_val_macro': [F1_val_macro]
                }
                )
        else:
            table = pd.concat([
                table, 
                pd.DataFrame(
                    {
                    'epoch': [epoch],
                    'loss': [loss], 
                    'val_loss': [val_loss], 
                    'F1_micro': [F1_micro], 
                    'F1_macro': [F1_macro],
                    'F1_val_micro': [F1_val_micro],
                    'F1_val_macro': [F1_val_macro]
                    }
                )
            ])
    else:
        if epoch == 0:
            table = pd.DataFrame({
                    'epoch': [epoch],
                    'loss': [loss], 
                    'F1_micro': [F1_micro], 
                    'F1_macro': [F1_macro]})
        else:
            table = pd.concat([
                table, 
                pd.DataFrame(
                    {
                    'epoch': [epoch],
                    'loss': [loss], 
                    'F1_micro': [F1_micro], 
                    'F1_macro': [F1_macro]
                    }
                )
            ])
    table.to_parquet(savepath+'/losses_per_epoch.parquet', engine='pyarrow')
    pass