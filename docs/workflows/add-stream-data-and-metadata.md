# Add stream data and metadata

A stream is a time-series recorded alongside the video, such as MED-PC operant
events. Metadata are grouping variables (treatment, sex, cohort) you tag recordings
with. This guide adds one of each.

## Before you start

[Add a recording](add-a-recording.md) to attach these to. The stream example uses a
MED-PC data file (`.txt`) and its program (`.MPC`).

## Add a stream

=== "GUI"

    Open the recording and, in the **Streams** section, click **Add stream**
    (highlighted).

    ![A recording detail page with the Add stream button highlighted.](../assets/workflows/add-stream-data/recording-full.png)

    Choose the data file, the **MED-PC (Med Associates)** importer and the `.MPC`
    program, then click **Import stream**.

    ![The Add stream dialog set to the MED-PC importer with its data and program files chosen.](../assets/workflows/add-stream-data/add-stream-dialog.png)

=== "CLI"

    ```
    detail recording add-stream --experiment ~/Documents/DetailExperiments/operant-study --recording boxA --name medpc --importer medpc --data PX22.txt --program Golden_Liq_SelfAdmin.MPC
    ```

## Tag metadata

=== "GUI"

    Open the experiment's **Recording information** page from the rail on the left
    (highlighted).

    ![The app sidebar with the Metadata node highlighted.](../assets/workflows/add-stream-data/metadata-node.png)

    Click **Edit**, use **Add column** for each grouping variable, fill each
    recording's row, then click **Save**. Or use **Import CSV** to set them in bulk.

    ![The Recording information table with treatment and sex columns for each recording.](../assets/workflows/add-stream-data/metadata-page.png)

=== "CLI"

    Set grouping variables in bulk from a CSV with a `recording_id` column (a `name`
    or video file name column works too):

    ```
    detail experiment import-metadata --experiment ~/Documents/DetailExperiments/operant-study --csv metadata.csv
    ```

    Or one recording at a time:

    ```
    detail experiment set-metadata --experiment ~/Documents/DetailExperiments/operant-study --recording boxA --set treatment=drug_A --set sex=M
    ```

## Next

- [Import recordings from YAML](import-recordings-from-yaml.md): create many recordings at once.
