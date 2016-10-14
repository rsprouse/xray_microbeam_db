# xray_microbeam_db
Tools for working with the X-Ray Microbeam Database.

Currently the only tool is a Bokeh app for visualizing data files.

# Getting started

    git clone https://github.com/rsprouse/xray_microbeam_db
    cd xray_microbeam_db/app
    ln -s /path/to/xray/files data
    bokeh serve xrayvis.py
    
Then visit http://localhost:5006/xrayvis in a web browser (Firefox preferred).

The file selection box could take a long time to populate if many files are in the data directory.



