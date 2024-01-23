"""
Collection of Functions used to visualize audio files in time domain.
Single/Multi File plotting with additional patient related information in one common plot or subplots.
Use Bokeh.
"""

from copy import deepcopy

import matplotlib as mpl
import numpy as np
import pandas as pd
import streamlit as st
from bokeh.layouts import column, row
from bokeh.models import CheckboxGroup, DataTable, TableColumn
from bokeh.models import CustomJS, Rect
from bokeh.models import RangeTool, Range1d, Legend, LegendItem
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.plotting import figure

from general_utilities import utilities


@st.cache_resource(show_spinner=False)
def plotting_single_file(current_file, name, opt, index):
    """ This functions returns a Bokeh figure of one single file with no additional information.

    :param current_file: Dataframe of wav-file with two columns ['time_steps', 'dim_0']
    :type current_file: pd.Dataframe
    :param name: Name of wav-file
    :type name: str
    :param opt: Options to display additional information
    :type opt: str
    :param index: Index of dataframe/audio file in session state
    :type index: int

    :return: Figure displayed with streamlit.bokeh_charts
    :type: Figure
    """

    # Audio widget
    y = np.array(st.session_state.wav_dataframes[index].iloc[:, 1])
    st.markdown('---')
    st.audio(y, sample_rate=st.session_state.librosa_sr)

    # Choose x-axis unit

    data = deepcopy(current_file)
    if opt == 'Artificial Timestamps':
        data.iloc[:, 0] = data.index
        x_name = 'Time in steps'
    else:
        x_name = 'Time in seconds'

    source = ColumnDataSource(data)

    # Main figure
    if max(np.abs(data.iloc[:, 1])) > 0.5:
        height = 1
    else:
        height = 0.5

    p = figure(height=400, width=1400, title=name, x_range=(data.iloc[0, 0], data.iloc[-1, 0]),
               y_range=(-height, height), tools="box_zoom, wheel_zoom, xpan,reset, save")
    line = p.line('time_steps', 'dim_0', source=source)
    p.yaxis.axis_label = 'Amplitude'
    p.xaxis.axis_label = x_name

    # Add hover tool to the main figure
    tooltips = [('Time Stamp', "@time_steps"), ('Amplitude', '@dim_0')]
    hover_tool = HoverTool(tooltips=tooltips, renderers=[line])
    p.add_tools(hover_tool)

    # Add range tool to the main figure and display additional plot under it
    select = figure(height=150, width=1400, title='Drag the selection box to change the range',
                    toolbar_location=None, tools="")
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = 'navy'
    range_tool.overlay.fill_alpha = 0.2
    select.line('time_steps', 'dim_0', source=source)
    select.x_range = Range1d(start=data.iloc[0, 0], end=data.iloc[-1, 0])
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool

    if opt == 'Respiratory Cycles':
        # Respiratory Cycles extraction
        az_df = st.session_state.wav_az_dict[index]
        color_map = {
            (0, 0): 'green',
            (1, 0): 'yellow',
            (0, 1): 'sandybrown',
            (1, 1): 'red'
        }
        coding_map = {
            (0, 0): 'N',
            (1, 0): 'C',
            (0, 1): 'W',
            (1, 1): 'B'
        }

        colors = []
        codings = ''

        ticks = [round(az_df.iloc[0, 0], 2)]   # extract all ticks to use for cycle start time

        for i, r in az_df.iterrows():
            start = r['start']
            end = r['end']
            ticks.append(round(end, 2))   # append cycle end time to the ticks' list

            crackles = r['crackles']
            wheezes = r['wheezes']
            color = color_map[(crackles, wheezes)]
            codings = codings + coding_map[(crackles, wheezes)] + " "
            colors.append(color)

            if max(np.abs(data.iloc[:, 1])) > 0.5:  # define rectangle height based on time series max value
                height = 1
            else:
                height = 0.5
            rec = Rect(x=start + (end - start) / 2, y=0, width=end - start, height=height * 2, fill_color=color,
                       fill_alpha=0.15, line_color='gray')
            p.add_glyph(rec)    # add a rectangle bordering the respiratory cycle, fill it in coded color

        p.xaxis.ticker = ticks  # add ticks
        p.xaxis.major_label_orientation = 45
        p.xgrid.ticker = ticks

        # Legend
        color_legend_dict = {
            'Category': ['White', 'Green', 'Yellow', 'Orange', 'Red', 'Nr. of R. Cycles', 'Presence/Cycle'],
            'Cycle Information': ['No Cycle detected', 'Normal - No Cr., No Wh.', 'Crackles', 'Wheezes', 'Both',
                                  az_df.shape[0], codings],
            'Abbreviation': [' ', 'N', 'C', 'W', 'B', ' ', ' ']
        }
        color_legend_df = pd.DataFrame(color_legend_dict)
        color_legend_source = ColumnDataSource(color_legend_df)
        color_legend_cols = [TableColumn(field=di, title=di) for di in color_legend_df.columns]
        color_legend_table = DataTable(source=color_legend_source, columns=color_legend_cols, width=550, height=200)

        color_legend_checkbox_group = CheckboxGroup(
            labels=["Color Legend for Respiratory Cycles"], active=[0])
        color_legend_checkbox_group.js_on_change('active',
                                                 CustomJS(args=dict(
                                                     color_legend_checkbox_group=color_legend_checkbox_group,
                                                     color_legend_table=color_legend_table),
                                                     code="""
                                                       if (color_legend_checkbox_group.active.length >0) {
                                                       color_legend_table.visible=true;
                                                       az_info_table=true;
                                                       } else {
                                                       color_legend_table.visible=false;
                                                       }
                                                       """))

        # Layout all
        color_legend_layout = column(color_legend_checkbox_group, color_legend_table)
        final_layout = column(color_legend_layout, p, select)
    else:
        final_layout = column(p, select)

    # Display all
    return final_layout


@st.cache_resource(show_spinner=False)
def plotting_single_file_patientinfo(current_file, name, opt, index):
    """ This functions returns a Bokeh figure of one single file and corresponding information to the patient
    if the user requests so.

    :param current_file: Dataframe of wav-file with two columns ['time_steps', 'dim_0']
    :type current_file: pd.Dataframe()
    :param name: Name of wav_lib-file
    :type name: str
    :param opt: Options to display additional information
    :type opt: str
    :param index: Index of dataframe/audio file in session state
    :type index: int

    :return: Figure displayed with streamlit.bokeh_charts
    :type: Figure
    """

    # Audio widget
    y = np.array(st.session_state.wav_dataframes[index].iloc[:, 1])
    st.markdown('---')
    st.audio(y, sample_rate=st.session_state.librosa_sr)

    # x-axis unit
    data = deepcopy(current_file)
    if opt == 'Artificial Timestamps':
        data.iloc[:, 0] = data.index
        x_name = 'Time in steps'
    else:
        x_name = 'Time in seconds'

    source = ColumnDataSource(data)

    # Main figure
    if max(np.abs(data.iloc[:, 1])) > 0.5:
        height = 1
    else:
        height = 0.5

    p = figure(height=400, width=1400, title=name, x_range=(data.iloc[0, 0], data.iloc[-1, 0]),
               y_range=(-height, height), tools="box_zoom, wheel_zoom, xpan,reset, save")
    line = p.line('time_steps', 'dim_0', source=source)
    p.yaxis.axis_label = 'Amplitude'
    p.xaxis.axis_label = x_name

    # Add hover tool to the main figure
    tooltips = [('Time Stamp', "@time_steps"), ('Amplitude', '@dim_0')]
    hover_tool = HoverTool(tooltips=tooltips, renderers=[line])
    p.add_tools(hover_tool)

    # Add range tool to the main figure and display additional plot under it
    select = figure(height=150, width=1400, title='Drag the selection box to change the range',
                    toolbar_location=None, tools="")
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = 'navy'
    range_tool.overlay.fill_alpha = 0.2
    select.line('time_steps', 'dim_0', source=source)
    select.x_range = Range1d(start=data.iloc[0, 0], end=data.iloc[-1, 0])
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool

    # Recording Information
    rec_data = utilities.get_rec_info_patient(index)
    rec_source = ColumnDataSource(rec_data)
    rec_cols = [TableColumn(field=di, title=di) for di in rec_data.columns]
    rec_table = DataTable(source=rec_source, columns=rec_cols, width=500, height=200)
    rec_checkbox_group = CheckboxGroup(labels=["Show Recording Information"], active=[0])
    rec_checkbox_group.js_on_change('active',
                                    CustomJS(args=dict(rec_checkbox_group=rec_checkbox_group, rec_table=rec_table),
                                             code="""
                                                       if (rec_checkbox_group.active.length >0) {
                                                       rec_table.visible=true;
                                                       } else {
                                                       rec_table.visible=false;
                                                       }
                                                       """))

    # Patient Information
    patient_data = utilities.get_pat_info_patient(index)
    pat_source = ColumnDataSource(patient_data)
    pat_cols = [TableColumn(field=di, title=di) for di in patient_data.columns]
    pat_table = DataTable(source=pat_source, columns=pat_cols, width=300, height=200)

    pat_checkbox_group = CheckboxGroup(labels=["Show Patient Information"], active=[0])
    pat_checkbox_group.js_on_change('active', CustomJS(args=dict(pat_checkbox_group=pat_checkbox_group),
                                                       code="""
                                                   if (pat_checkbox_group.active.length >0) {
                                                   pat_table.visible=true;
                                                   } else {
                                                   pat_table.visible=false;
                                                   }
                                                   """))
    if opt == 'Respiratory Cycles':
        az_df = st.session_state.wav_az_dict[index]
        color_map = {
            (0, 0): 'green',
            (1, 0): 'yellow',
            (0, 1): 'sandybrown',
            (1, 1): 'red'
        }
        coding_map = {
            (0, 0): 'N',
            (1, 0): 'C',
            (0, 1): 'W',
            (1, 1): 'B'
        }
        ticks = [round(az_df.iloc[0, 0], 2)]    # Extract start times for respiratory cycles
        colors = []
        codings = ''
        for i, r in az_df.iterrows():
            start = r['start']
            end = r['end']
            ticks.append(round(end, 2))     # Append end times for respiratory cycles

            crackles = r['crackles']
            wheezes = r['wheezes']
            color = color_map[(crackles, wheezes)]
            codings = codings + coding_map[(crackles, wheezes)] + " "
            colors.append(color)

            if max(np.abs(data.iloc[:, 1])) > 0.5:      # Choose rectangular height based on time series max value
                height = 1
            else:
                height = 0.5
            rec = Rect(x=start + (end - start) / 2, y=0, width=end - start, height=height * 2, fill_color=color,
                       fill_alpha=0.15, line_color='gray')
            p.add_glyph(rec)        # Add rectangle bordering the respiratory cycle

        p.xaxis.ticker = ticks      # Add ticks
        p.xaxis.major_label_orientation = 45
        p.xgrid.ticker = ticks

        # Legend
        color_legend_dict = {
            'Category': ['White', 'Green', 'Yellow', 'Orange', 'Red', 'Nr. of R. Cycles', 'Presence/Cycle'],
            'Cycle Information': ['No Cycle detected', 'Normal - No Cr., No Wh.', 'Crackles', 'Wheezes', 'Both',
                                  az_df.shape[0], codings],
            'Abbreviation': [' ', 'N', 'C', 'W', 'B', ' ', ' ']
        }
        color_legend_df = pd.DataFrame(color_legend_dict)
        color_legend_source = ColumnDataSource(color_legend_df)
        color_legend_cols = [TableColumn(field=di, title=di) for di in color_legend_df.columns]
        color_legend_table = DataTable(source=color_legend_source, columns=color_legend_cols, width=550, height=200)

        color_legend_checkbox_group = CheckboxGroup(labels=["Color Legend for Respiratory Cycles"], active=[0])
        color_legend_checkbox_group.js_on_change('active',
                                                 CustomJS(args=dict(
                                                     color_legend_checkbox_group=color_legend_checkbox_group,
                                                     color_legend_table=color_legend_table),
                                                     code="""
                                                   if (color_legend_checkbox_group.active.length > 0) {
                                                   color_legend_table.visible=true;
                                                   } else {
                                                   color_legend_table.visible=false;
                                                   }
                                                   """))
        # Layout all
        rec_layout = column(rec_checkbox_group, rec_table)
        pat_layout = column(pat_checkbox_group, pat_table)
        color_legend_layout = column(color_legend_checkbox_group, color_legend_table)
        info_layout = row(rec_layout, pat_layout, color_legend_layout)
        final_layout = column(info_layout, p, select)
    else:
        rec_layout = column(rec_checkbox_group, rec_table)
        pat_layout = column(pat_checkbox_group, pat_table)
        info_layout = row(rec_layout, pat_layout)
        final_layout = column(info_layout, p, select)

    # Display all
    return final_layout


def _plot_single_file(data, file, opt, index):
    if opt == 'Artificial Timestamps':
        data.iloc[:, 0] = data.index
        x_name = 'Time in steps'
    else:
        x_name = 'Time in seconds'

    source = ColumnDataSource(data)

    # Main figure
    p = figure(height=350, width=700, title=file, x_range=(data.iloc[0, 0], data.iloc[-1, 0]))
    line = p.line('time_steps', 'dim_0', source=source)
    p.yaxis.axis_label = 'Amplitude'
    p.xaxis.axis_label = x_name

    # Add hover tool to the main figure
    tooltips = [('Time Stamp', "@time_steps"), ('Amplitude', '@dim_0')]
    hover_tool = HoverTool(tooltips=tooltips, renderers=[line])
    p.add_tools(hover_tool)

    #  Respiratory Cycles
    if opt == 'Respiratory Cycles':
        az_df = st.session_state.wav_az_dict[index]
        color_map = {
            (0, 0): 'green',
            (1, 0): 'yellow',
            (0, 1): 'sandybrown',
            (1, 1): 'red'
        }

        ticks = [round(az_df.iloc[0, 0], 2)]    # Extract start times for respiratory cycles
        colors = []
        for j, r in az_df.iterrows():
            start = r['start']
            end = r['end']
            ticks.append(round(end, 2))    # Extract end times for respiratory cycles

            crackles = r['crackles']
            wheezes = r['wheezes']
            color = color_map[(crackles, wheezes)]
            colors.append(color)
            if max(np.abs(data.iloc[:, 1])) > 0.5:      # Choose rectangular height based on time series max value
                height = 1
            else:
                height = 0.5
            rec = Rect(x=start + (end - start) / 2, y=0, width=end - start, height=height * 2, fill_color=color,
                       fill_alpha=0.15, line_color='gray')
            p.add_glyph(rec)        # Add rectangle bordering the respiratory cycle

        p.xaxis.ticker = ticks      # Add ticks
        p.xaxis.major_label_orientation = 45
        p.xgrid.ticker = ticks

    return p


@st.cache_resource(show_spinner=False)
def all_files_in_subplots(df_list, file_names_list, opt):
    """
    Plot all files in separate subplots - two files per row.
    :param df_list: List of dataframes
    :type df_list: list
    :param file_names_list: List of names corresponding to each file
    :type file_names_list: list
    :param opt: 'Time in Seconds', 'Artificial Timestamps', 'Respiratory Cycles' - determines what units to use for
                the x-axis. 'Respiratory Cycles' will display each beginning and end of a respiratory cycle.
    :type opt: List
    """
    for i in range(0, len(df_list), 2):  # Loop with step 2 to process two files at a time
        cols = st.columns(2)
        # Data and corresponding figure 1
        data1 = deepcopy(df_list[i])
        plot1 = _plot_single_file(data1, file_names_list[i], opt, i)
        # Audio widget 1
        y1 = np.array(st.session_state.wav_dataframes[i].iloc[:, 1])
        cols[0].audio(y1, sample_rate=st.session_state.librosa_sr)

        if i + 1 < len(df_list):        # If a second file is plotted in the same row
            data2 = deepcopy(df_list[i + 1])
            plot2 = _plot_single_file(data2, file_names_list[i + 1], opt, i + 1)

            y2 = np.array(st.session_state.wav_dataframes[i + 1].iloc[:, 1])
            cols[1].audio(y2, sample_rate=st.session_state.librosa_sr)

            st.bokeh_chart(row(plot1, plot2))
        else:       # Display only the first plot
            st.bokeh_chart(plot1)
        st.markdown('---')


@st.cache_resource(show_spinner=False)
def all_files_in_one_plot(df_list, file_names_list, opt):
    """"
    Plot all files in one common plot.
    :param df_list: List of files to be plotted in a dataframe format
    :type df_list: list
    :param file_names_list: List of names corresponding to each file
    :type file_names_list: list
    :param opt: 'Time in Seconds', 'Artificial Timestamps' - determines what units to use for
                the x-axis.
    :type opt: List
    """
    p = figure(height=400, width=1400, tools="box_zoom, wheel_zoom, xpan,reset, save")

    # Prep Colors
    my_colors = list(np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(df_list), replace=False))

    # Legend preparation
    legend_items = []

    # Loop through all multiselect categories to add a trace to the plot
    for i in range(len(df_list)):
        color = my_colors[i % len(my_colors)]

        # get filename from index dictionary
        file = file_names_list[i]
        # get the current file to add to the plot and sort by timestamp column
        current_file = df_list[i]
        data = deepcopy(current_file)

        if opt == 'Artificial Timestamps':
            data.iloc[:, 0] = data.index
            x_name = 'Time in steps'
        else:
            x_name = 'Time in seconds'

        source = ColumnDataSource(data)

        # Add to the main figure
        ts = p.line('time_steps', 'dim_0', source=source, line_width=1.5, color=mpl.colors.XKCD_COLORS[color])
        p.xaxis.axis_label = x_name
        legend_items.append(LegendItem(label=file, renderers=[ts]))

        # Add hover tool to line of timeseries ts
        tooltips = [('File', f'{file}'), ('Time Stamp', "@time_steps"), ('Amplitude', '@dim_0')]
        hover_tool = HoverTool(tooltips=tooltips, renderers=[ts])
        p.add_tools(hover_tool)

    p.title = 'All files in one plot'
    p.yaxis.axis_label = 'Amplitude'
    p.x_range.start = 0
    legend = Legend(items=legend_items, click_policy='hide')
    p.add_layout(legend, 'right')
    st.bokeh_chart(p)


@st.cache_resource(show_spinner=False)
def multiple_files_in_subplots(df_list, file_names_list, file_multi_chooser, opt):
    """
    Plot multiple files in separate subplots - two files per row.
    :param df_list: List of files to be plotted in dataframe format
    :type df_list: list
    :param file_names_list: List of names corresponding to each file
    :type file_names_list: list
    :param file_multi_chooser: User selection obtained with st.multiselect
    :type file_multi_chooser: list
    :param opt: 'Time in Seconds', 'Artificial Timestamps', 'Respiratory Cycles' - determines what units to use for
                the x-axis. 'Respiratory Cycles' will display each beginning and end of a respiratory cycle.
    :type opt: List
    """
    for i in range(0, len(file_multi_chooser), 2):
        cols = st.columns(2)
        # get the current file to add to the plot and sort by timestamp column
        file = file_multi_chooser[i]
        index = file_names_list.index(file)
        data = deepcopy(df_list[index])
        plot1 = _plot_single_file(data, file_names_list[index], opt, index)

        # Audio widget
        y1 = np.array(st.session_state.wav_dataframes[index].iloc[:, 1])
        cols[0].audio(y1, sample_rate=st.session_state.librosa_sr)

        if i + 1 < len(file_multi_chooser):
            data2 = deepcopy(df_list[index + 1])
            plot2 = _plot_single_file(data2, file_names_list[index + 1], opt, index + 1)

            y2 = np.array(st.session_state.wav_dataframes[index + 1].iloc[:, 1])
            cols[1].audio(y2, sample_rate=st.session_state.librosa_sr)

            st.bokeh_chart(row(plot1, plot2))
        else:
            st.bokeh_chart(plot1)


@st.cache_resource(show_spinner=False)
def multiple_files_in_one_plot(df_list, file_names_list, file_multi_chooser, opt):
    """
    Plot multiple files in one common plot.
    :param df_list: List of files to be plotted in a dataframe format
    :type df_list: list
    :param file_names_list: List of names corresponding to each file
    :type file_names_list: list
    :param file_multi_chooser: User selection obtained with st.multiselect
    :type file_multi_chooser: list
    :param opt: 'Time in Seconds', 'Artificial Timestamps' - determines what units to use for the x-axis.
    :type opt: List
    """
    p = figure(height=400, width=1400, tools="box_zoom, wheel_zoom, xpan,reset, save")
    # Prep Colors
    my_colors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(df_list), replace=False)

    # Legend preparation
    legend_items = []

    # Loop through all files in file_multi_chooser
    for i, file in enumerate(file_multi_chooser):

        color = my_colors[i % len(my_colors)]

        # extract file name and the current file and sort by timestamp column
        index = file_names_list.index(file)
        current_file = df_list[index]

        data = deepcopy(current_file)

        if opt == 'Artificial Timestamps':
            data.iloc[:, 0] = data.index
            x_name = 'Time in steps'
        else:
            x_name = 'Time in seconds'

        source = ColumnDataSource(data)

        # Main figure
        ts = p.line('time_steps', 'dim_0', source=source, line_width=1.5, color=mpl.colors.XKCD_COLORS[color])
        p.xaxis.axis_label = x_name
        legend_items.append(LegendItem(label=file, renderers=[ts]))

        # Add hover tool to line of timeseries ts
        tooltips = [('File', f'{file}'), ('Time Stamp', "@time_steps"), ('Amplitude', '@dim_0')]
        hover_tool = HoverTool(tooltips=tooltips, renderers=[ts])
        p.add_tools(hover_tool)

    p.yaxis.axis_label = 'Amplitude'
    p.x_range.start = 0
    p.title = 'Multiple files in one plot'
    legend = Legend(items=legend_items, click_policy='hide')
    p.add_layout(legend, 'right')
    st.bokeh_chart(p)
