import subprocess
def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def downloadScratchRemoverModel():
    runcmd("cd extensions", verbose=True)
    runcmd("cd arifScratchRemoverWebUIExtention", verbose=True)
    runcmd("wget https://www.dropbox.com/s/5jencqq4h59fbtb/FT_Epoch_latest.pt", verbose=True)
    runcmd("cd ..", verbose=True)
    runcmd("cd ..", verbose=True)


#runcmd("apt-get update && apt-get install libgl1", verbose = True)

