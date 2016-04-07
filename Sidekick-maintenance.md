# Sidekick - Maintenance

## Crawler

### Check the crawler status
Connect to the server with:

````
ssh vetter@icsil1-prj-01.epfl.ch
````

There four scripts that are running:

- `crawler/projectsScraper.py`
- `crawler/updateProjectsStatusSingle.py`
- `crawler/twitterWatcher.py`
- `mining/predict_live_projects.py`
 
Watch the log console for these four scripts with:
 
````
screen -R
````
 
This will open a [Screen](https://developer.apple.com/library/mac/documentation/Darwin/Reference/ManPages/man1/screen.1.html) window with four tabs. To navigate between tabs, use `Ctrl-Z n` or `Ctrl <tab number>` where `<tab number>` is 0, 1, 2 or 3. You can stop a script with `Ctrl-C`. 
 
The activate the virtual environment, do:

````
. ~/venv/bin/activate
```` 
 
You can now launch the scripts again with:

- `python projectsScraper.py`
- `python updateProjectsStatusSingle.py`
- `python twitterWatcher.py`
- `python predict_live_projects.py /home/90days/vetter/kickstarter/models/2013.09.24`

To run the last command, the folder `/home/90days/` must be mounted. To do so, you can try the following commands:

````
## Reconnect the folder
sshfs -o reconnect kristof@icsil1-access2.epfl.ch:/home/90days /home/90days
## Remount  the folder from scratch
ps -ef | grep -w sshfs                                          # Find the process
kill -9 [pid-sshfs-process]                                     # Kill it
fusermount -u /home/90days                                      # Unmount the folder
sshfs kristof@icsil1-access2.epfl.ch:/home/90days /home/90days  # Mount the folder
````
