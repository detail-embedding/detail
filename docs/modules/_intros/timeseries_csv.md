<!-- Prepended (prose only, no heading) to the generated modules/timeseries_csv.md page. -->
Imports a CSV of continuous measurements as a stream: one time column and one or more numeric
channels. Leave the parameters below unset and the importer reads a file already shaped for
DETAIL, where the time column is named `seconds`, `frame` or `t_ns` and every other column
becomes a channel. Set them and it reads any CSV, under whatever names the source tool used.

One CSV makes one stream carrying the channels you name. The alignment offset belongs to the
stream, and channels recorded by one device share a clock, so keeping them together means
syncing once rather than once per column.
