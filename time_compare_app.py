# EPI-USE Intern Project by Alec Han
# Please refer to the functional process document for additional information such as Methodology, Process Diagram, Assumptions, and Test Cases 

# Import packages needed for this application
import sys
import subprocess
import importlib

# Function to install and import a package
def install_and_import(package):
    try:
        return importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(package)

# Import required packages (automatically installing if missing)
np = install_and_import('numpy')
pd = install_and_import('pandas')
st = install_and_import('streamlit')


# Session states are used to remember user progress. Session state is a special Streamlit object that PERSISTS variables across reruns of the application. 
# Every time the user interacts with something (uploads a file, clicks a button, changes an input), Streamlit RERUNS the script top to bottom.
# Without session state, you can lose all intermediate progress. 
if 'upload_complete' not in st.session_state:
    # Marks whether the user has uploaded both files and clicked "Confirm Files and Threshold" button.
    # Only when this is True, will the script proceed with preprocessing and mapping setup.
    st.session_state.upload_complete = False
if 'mapping_complete' not in st.session_state:
    # Marks whether the activity/pay code mapping has been confirmed.
    # Only when this is True, will the user be allowed to run the comparison function.
    st.session_state.mapping_complete = False
if 'activity_mapping' not in st.session_state:
    # Stores the dictionary of mappings.
    # Pass this into the compare_time function. 
    st.session_state.activity_mapping = None

# Display the app title at the top of the browser page.
st.title("Time Compare Tool")

# File uploaders
# Creates a file uploader widget for the legacy system dataset and for the new system (WorkForce Software)
legacy_file = st.file_uploader("Upload Legacy System CSV/XLSX:", type=['csv', 'xlsx'])
new_file = st.file_uploader("Upload WFS System CSV/XLSX:", type=['csv','xlsx'])

# Creates a number input widget where the user can enter a threshold for acceptable time differences (in percentage)
# Set minimum value to 0 to prevent negative numbers inputted and set default value to 98%
threshold = st.number_input("Enter threshold to define a mismatch for time worked (Example: 0.98 for 98%):", min_value=0.0, value=0.98)

if legacy_file and new_file:
    st.success("Files successfully uploaded.")
    if st.button("Confirm Files and Threshold"):
        st.session_state.upload_complete = True
    

# Creates another number input widget where the user can set how many top mismatches they want to see.
# Minimum is 1 and default is 10
# I am commenting out this input for now as I do not see the value in only seeing the top X mismatches. 
# top_x = st.number_input("Enter the number of top mismatches to display", min_value=1, value=10)


def compare_time(legacy_df, wfs_df, threshold, activity_mapping, column_pairs, num_records_legacy):
    # Build combined_df.
    combined_df = pd.concat([legacy_df, wfs_df], axis=1)

    # Compare all the columns (including handling NaN and thresholds).
    # This method directly compares two Pandas Series (representing the columns). It returns True only if both Series have the same 
    # shape, elements, and order, including handling NaN values as equal if they are in the same position.
    st.subheader("Numeric and Time Column Comparisons")
    num_of_mismatches = 0
    
    for col1, col2 in column_pairs:

        # For debugging purposes. Comment out when not needed. 
        # st.write("Legacy column dtype:", col1, legacy_df[col1].dtype)
        # st.write("WFS column dtype:", col2, wfs_df[col2].dtype)

        are_equal = legacy_df[col1].equals(wfs_df[col2])
        st.write(f"Are `{col1}` and `{col2}` equal? {are_equal}")
        if are_equal == False:
            # DEBUG STATEMENTS - Check data type
            if legacy_df[col1].dtype != wfs_df[col2].dtype:
                st.write(f"Type mismatch: {legacy_df[col1].dtype} vs {wfs_df[col2].dtype}")
            # DEBUG STATEMENTS - Check shape
            if legacy_df[col1].shape != wfs_df[col2].shape:
                st.write(f"Shape mismatch: {legacy_df[col1].shape} vs {wfs_df[col2].shape}")


            # This if statement is particularly for the DURATION_MINUTES columns 
            if np.issubdtype(legacy_df[col1].dtype, np.timedelta64):
                # Convert time deltas to float minutes for comparison
                legacy_minutes = legacy_df[col1] / pd.Timedelta(minutes=1)
                wfs_minutes = wfs_df[col2] / pd.Timedelta(minutes=1)
                rel_diff = abs(legacy_minutes - wfs_minutes) / legacy_minutes.replace(0, np.nan)
                differences = (rel_diff > (1 - threshold)) | (legacy_minutes.isna() ^ wfs_minutes.isna())
                # differences = ~np.isclose(legacy_minutes, wfs_minutes, rtol=(1 - threshold), equal_nan=True)
                
                # For debugging:
                # st.write("Legacy minutes sample:", legacy_minutes[differences].head())
                # st.write("WFS minutes sample:", wfs_minutes[differences].head())

                legacy_vals = legacy_df.loc[differences, col1]
                wfs_vals = wfs_df.loc[differences, col2]
                difference_in_values = abs(legacy_vals - wfs_vals)
                
                # Clean the display by removing converting the values to strings and removing the days. 
                legacy_vals = legacy_vals.astype(str).str.replace("0 days ", "", regex=False)
                wfs_vals = wfs_vals.astype(str).str.replace("0 days ", "", regex=False)
                difference_in_values = difference_in_values.astype(str).str.replace("0 days ", "", regex=False)

                mismatches = pd.DataFrame({
                    'legacy_value': legacy_vals,
                    'wfs_value': wfs_vals,
                    'Difference':difference_in_values})
    

            elif np.issubdtype(legacy_df[col1].dtype, np.number):
                # Use isclose for floating point comparisons with a threshold
                differences = ~np.isclose(legacy_df[col1], wfs_df[col2], rtol=(1-threshold), equal_nan=True)  # threshold is the user allowed difference 

                # Calculate the max difference to determine there the difference is from floating point precision.
                num_differences = differences.sum()
                max_diff = np.abs(legacy_df[col1] - wfs_df[col2]).max()

                if num_differences == 0:
                    st.write(f"Columns {col1} and {col2} are not strictly equal (per .equals), but all differences are due to floating point precision (max diff: {max_diff:.3e}).")

                # DEBUG PRINTS: show first few mismatched values and max difference
                # st.write(f"\n--- FLOATING POINT MISMATCH DEBUG: {col1} vs {col2} ---")
                # st.write("Legacy mismatched values:\n", legacy_df.loc[differences, col1].head(10))
                # st.write("WFS mismatched values:\n", wfs_df.loc[differences, col2].head(10))
                # st.write("Max absolute difference:\n", np.abs(legacy_df[col1] - wfs_df[col2]).max())

                legacy_vals = legacy_df.loc[differences, col1]
                wfs_vals = wfs_df.loc[differences, col2]

                mismatches = pd.DataFrame({
                    'legacy_value': legacy_vals,
                    'wfs_value': wfs_vals})
                mismatches['Difference'] = abs(mismatches['legacy_value']-mismatches['wfs_value'])

            else:
                # Element-wise comparison that treats NaNs in the same location as equal
                differences = ~((legacy_df[col1] == wfs_df[col2]) | (legacy_df[col1].isna() & wfs_df[col2].isna()))
                legacy_vals = legacy_df.loc[differences, col1]
                wfs_vals = wfs_df.loc[differences, col2]

                mismatches = pd.DataFrame({
                    'legacy_value': legacy_vals,
                    'wfs_value': wfs_vals})

            num_of_mismatches += len(mismatches)
            st.write(f"**There are {len(mismatches)} mismatches between `{col1}` and `{col2}`**")
            st.write(f"**Mismatches between `{col1}` and `{col2}`:**")
            st.dataframe(mismatches)

        st.divider()

    # Check mapping validation
    def is_valid_mapping(legacy_code, wfs_code, mapping_dict):
        if legacy_code == 'OTHER':
            # if the code is other, then it allows any pay code (wfs_code)
            return wfs_code in mapping_dict['OTHER']
        # if legacy_code (activity_code_legacy) is NOT other, check if the WFS pay code equals the one in the current row
        return mapping_dict.get(legacy_code) == wfs_code


    # This checks if the pair exists in the unique mapping list.
    # A new column called 'is_match' is created that holds true/false values for the comparison.
    combined_df['is_match'] = combined_df.apply(
        lambda row: is_valid_mapping(row['ACTIVITY_CODE_legacy'], row['PAY_CODE_wfs'], activity_mapping), axis=1)


    # Are all the rows correctly matched according to the mapping table?
    comparison_result = combined_df['is_match'].all()

    # Add the number of mismatches from the results of comparing activity codes to pay codes. 
    num_of_mismatches += combined_df['is_match'].value_counts().get(False, 0)


    st.write("### Comparing activity code and pay code columns:")
    st.write("Do all rows match the expected activity/pay code mapping?", comparison_result)
    st.write(f"There are {combined_df['is_match'].value_counts().get(False, 0)} mismatches found in activity codes to pay codes.")

    # Show mismatches for manual review
    mismatches = combined_df[~combined_df['is_match']]

    # Display mismatches (if any)
    if not mismatches.empty:
        st.write("### Mismatches found in activity/pay code mapping:")
        st.dataframe(mismatches[['ACTIVITY_CODE_legacy', 'PAY_CODE_wfs']])
    else:
        st.success("All activity/pay code mappings matched correctly.")

    percent_match = (1 - (num_of_mismatches / num_records_legacy)) * 100
    st.subheader("Summary of Key Statistics:")
    st.markdown(f"""
                - **Total Records Compared:** {num_records_legacy}
                - **Total Mismatches:** {num_of_mismatches}
                - **% Match Accuracy:** **{percent_match:.2f}%**
                - **Threshold Used:** {threshold * 100:.2f}% (i.e., ≥ {threshold * 100:.2f}% similarity)
                - **Activity Code / Pay Code Match:** {'All matched' if comparison_result else f'{len(mismatches)} mismatches found'}
                """)



# Process the files after both are uploaded
if st.session_state.upload_complete:
    
    # SECTION 1: VARIABLE CREATION --------------------------------------------------------------------------------------------------
    if legacy_file.name.endswith('.csv'):
        legacy_df_original = pd.read_csv(legacy_file)
    else:
        legacy_df_original = pd.read_excel(legacy_file)
    
    if new_file.name.endswith('.csv'):
        wfs_df_original = pd.read_csv(new_file)
    else:
        wfs_df_original = pd.read_excel(new_file)

    # Check the length of each dataframe to verify one is not an incomplete file or one does not contain extra records.
    num_records_legacy = len(legacy_df_original)
    num_records_wfs = len(wfs_df_original)
    if num_records_legacy != num_records_wfs:
        st.write("The number of records do not match between the legacy file and the WFS file.")
        st.write(f"There are **{num_records_legacy}** records in the legacy file and **{num_records_wfs}** records in the WFS file.")
        st.error("Please review the datasets uploaded prior to continuing.")
        st.stop()
    else:
        st.write(f"There are **{num_records_legacy}** records in the legacy file and **{num_records_wfs}** records in the WFS file.")
        st.success("The length of each data set match.")

    # Display a preview of the uploaded data sets to allow the user another opportunity to verify the data looks correct. 
    st.subheader("Preview of Uploaded Files")
    st.write("Legacy Data Set:")
    st.dataframe(legacy_df_original.head(10))
    st.write("WFS Data Set:")
    st.dataframe(wfs_df_original.head(10))

    # Create copies of the uploaded data sets to allow us to modify the copies and not the original inputs. 
    # This ensures we retain an original copy that is un-modified in case we need to review/use them again. 
    legacy_df = legacy_df_original.copy()
    wfs_df = wfs_df_original.copy()

    # SECTION 2: PREPROCESSING ------------------------------------------------------------------------------------------------------
    # I am breaking up the START_DTTM and END_DTTM for WFS into separate columns to match the legacy system. 
    # This will also help with calculating work time (subtract end and start times for each record).

    # Creating new columns in the WFS DF breaking up the date/time into separate columns for both start and end. 
    # I'm also converting the columns to datetime64 so I can perform math on them while calculating the DURATION_MINUTES.
    # For the time columns, you must specify the format if we were keeping them in datetime64 format, instead I am converting them to timedelta objects.
    # datetime64 objects will automatically add a default date of 1/1/1900 for time data whileas timedelta allows time to be by itself. 
    wfs_df['START_DATE'] = pd.to_datetime(wfs_df['START_DTTM'].dt.date)
    wfs_df['START_TIME'] = pd.to_timedelta(wfs_df['START_DTTM'].dt.strftime('%H:%M:%S'))
    wfs_df['END_DATE'] = pd.to_datetime(wfs_df['END_DTTM'].dt.date)
    wfs_df['END_TIME'] = pd.to_timedelta(wfs_df['END_DTTM'].dt.strftime('%H:%M:%S'))

    # Remove the START_DTTM and END_DTTM columns now that I've extracted the date and time information and created new separate columns.
    wfs_df.drop(['START_DTTM','END_DTTM'],axis=1, inplace=True)
 
    # Create a new column called "DURATION_MINUTES" matching the legacy system naming.
    # With that column, subtract the end time and start time and convert to minutes.
    wfs_df['DURATION_MINUTES'] = (wfs_df['END_TIME'] - wfs_df['START_TIME']) / pd.Timedelta(minutes=1)
    # Fill the NaN values in the DURATION_MINUTES column with correct values by taking the values in the HOUR column and converting them to minutes
    wfs_df['DURATION_MINUTES'] = wfs_df['DURATION_MINUTES'].fillna(wfs_df['HOURS'] * 60)
    # Remove the HOURS column.
    wfs_df.drop('HOURS', axis=1, inplace=True)

    # Print statements to check status of dataframes. 
    # st.dataframe(legacy_df.head(10))
    # st.dataframe(wfs_df.head(10))

    # Display a preview of the uploaded data sets to allow the user another opportunity to verify the data looks correct. 
    # st.subheader("Uploaded Files After Pre-Processing is Complete")
    # st.write("Pre-Processing Legacy Dataset:")
    # st.dataframe(legacy_df.head(10))
    # st.write("Pre-Processing WFS Dataset:")
    # st.dataframe(wfs_df.head(10))


    # Rename the WFS PAY_CODE to ATIVITY_CODE to match legacy system. 
    # Commenting out because I want the columns to keep their distinct names. 
    # wfs_df = wfs_df.rename(columns={'PAY_CODE': 'ACTIVITY_CODE'})

    # Create variable that is a list of all the column names. 
    original_cols = legacy_df.columns.tolist() + wfs_df.columns.tolist()

    # Remove duplicates using set data type from the ORIGINAL column names.
    prefixes = set(original_cols)
    # print('prefixes:', prefixes)
    # {'ACTIVITY_CODE', 'END_DATE', 'END_TIME', 'DURATION_MINUTES', 'START_TIME', 'PAY_CODE', 'WORK_DATE', 'START_DATE', 'EMPLOYEE_ID'}
    # I am removing ACTIVITY_CODE and PAY_CODE because I do not want it to be a part of the comparison from the .equals code block below.
    # The activity/pay codes have their own method of comparison. 
    prefixes.remove("ACTIVITY_CODE")
    prefixes.remove("PAY_CODE")

    # Add a suffix for each corresponding data set's columns to separate what is legacy and what is WFS.
    legacy_df.columns = legacy_df.columns.map(lambda x: str(x) + '_legacy')
    wfs_df.columns = wfs_df.columns.map(lambda x: str(x) + '_wfs')

    # Convert the date related columns in the legacy system to datetime64. 
    legacy_date_columns = ['WORK_DATE_legacy','START_DATE_legacy','END_DATE_legacy']
    for col in legacy_date_columns:
        legacy_df[col] = pd.to_datetime(legacy_df[col])

    # Convert the time related columns in the legacy system to timedelta objects.
    legacy_time_columns = ['START_TIME_legacy','END_TIME_legacy']
    for col in legacy_time_columns:
        legacy_df[col] = pd.to_timedelta(legacy_df[col])

    # Create a new df that merges/concats the two separate df's. 
    combined_df = pd.concat([legacy_df, wfs_df], axis=1)
    # st.dataframe(combined_df.head())
    # st.write(combined_df.dtypes) # The employee_id_wfs was converted to int64 removing the leading zeroes. 


    # Create a list of all the column names 
    cols = combined_df.columns.tolist()
    # print('\ncols:', cols)
    # ['EMPLOYEE_ID_legacy', 'WORK_DATE_legacy', 'START_DATE_legacy', 'START_TIME_legacy', 'END_DATE_legacy', 'END_TIME_legacy', 'ACTIVITY_CODE_legacy', 'DURATION_MINUTES_legacy', 
    # 'EMPLOYEE_ID_wfs', 'WORK_DATE_wfs', 'PAY_CODE_wfs', 'START_DATE_wfs', 'START_TIME_wfs', 'END_DATE_wfs', 'END_TIME_wfs', 'DURATION_MINUTES_wfs']

    # Create a helper function that extracts the prefix and makes a pair of each column between legacy and WFS
    def generate_column_pairs(prefix, cols):

        # Create a generator that yields elements starting with the prefix
        matching_elements_iterator = (item for item in cols if item.startswith(prefix))
        
        pair = []
        for element in matching_elements_iterator:
            pair.append(element)

        return tuple(pair)
    
    column_pairs = []
    for prefix in prefixes:
        pair = generate_column_pairs(prefix, cols)
        column_pairs.append(pair)
    # print('column pairs:', column_pairs)
    # [('END_TIME_legacy', 'END_TIME_wfs'), ('DURATION_MINUTES_legacy', 'DURATION_MINUTES_wfs'), 
    # ('END_DATE_legacy', 'END_DATE_wfs'), ('START_TIME_legacy', 'START_TIME_wfs'), ('START_DATE_legacy', 'START_DATE_wfs'), 
    # ('EMPLOYEE_ID_legacy', 'EMPLOYEE_ID_wfs'), ('WORK_DATE_legacy', 'WORK_DATE_wfs')]


    # SECTION 3: SHOW USER FORM TO VERIFY/EDIT MAPPINGS -----------------------------------------------------------------------------
    # Find unique codes
    unique_legacy_codes = combined_df['ACTIVITY_CODE_legacy'].unique()
    unique_wfs_codes = combined_df['PAY_CODE_wfs'].unique()

    # Display the results for user to review.
    st.write("Unique Legacy Activity Codes:", unique_legacy_codes)
    st.write("Unique WFS Pay Codes:", unique_wfs_codes)

    # st.radio is a Streamlist widget that displays a list of radio buttons, where the user can select only 1 option.
    # Radio button to let user to select how they want to input the mapping
    mapping_input_method = st.radio("How would you like to provide the activity to pay code mapping?",
                        ("Manual Input","Upload Mapping File"))

    # Initialize activity_mapping
    activity_mapping = None

    if mapping_input_method == "Upload Mapping File":
        # File uploader for CSV or Excel files (supports multiple Excel formats)
        mapping_file = st.file_uploader("Upload CSV or Excel file with activity/pay code mapping.", type=['csv','xlsx','xls','xlsm','xlsb','odf','ods','odt'])

        if mapping_file:
            # Automatically use the appropriate pandas function based on file extension
            if mapping_file.name.endswith('.csv'):
                mapping_df = pd.read_csv(mapping_file)
            else:
                mapping_df = pd.read_excel(mapping_file)

            st.write("Uploaded Mapping Table:")
            # Display the uploaded file for user confirmation
            st.dataframe(mapping_df)
        
            # Validate required columns exists in the uploaded file
            if {'ACTIVITY_CODE_legacy','PAY_CODE_wfs'}.issubset(mapping_df.columns):
                # Build mapping directory
                activity_mapping = {}
                for legacy_code in mapping_df['ACTIVITY_CODE_legacy'].unique():
                    mapped_codes = mapping_df[mapping_df['ACTIVITY_CODE_legacy'] == legacy_code]['PAY_CODE_wfs'].tolist()
                    if legacy_code == 'OTHER':
                        # OTHER can map to multiple pay codes
                        activity_mapping[legacy_code] = mapped_codes
                    else:
                        # For non-'OTHER' codes, except only one mapping
                        activity_mapping[legacy_code] = mapped_codes[0]

                st.success("Mapping file successfully loaded and parsed.") 

                if st.button("Confirm Mapping File"):
                    st.session_state.activity_mapping = activity_mapping
                    st.session_state.mapping_complete = True
                    st.success("Mapping confirmed.")

            else:
                st.error("Mapping file must contain 'ACTIVITY_CODE_legacy' and PAY_CODE_wfs columns.")

    elif mapping_input_method == "Manual Input":

        st.write("Please define the expected mapping between legacy activity codes and WFS pay codes:")

        # Create a form to input the mappings
        with st.form("mapping_form"):
            activity_mapping = {}
            for legacy_code in unique_legacy_codes:
                if legacy_code == 'OTHER':
                    # For 'OTHER', allow multi-select since it maps to many codes
                    selected_codes = st.multiselect(f"Select WFS pay codes that match legacy code '{legacy_code}':", unique_wfs_codes)
                    activity_mapping[legacy_code] = selected_codes
                else:
                    selected_code = st.selectbox(f"Select WFS pay code that matches legacy code '{legacy_code}':", unique_wfs_codes)
                    activity_mapping[legacy_code] = selected_code

            submit_mapping = st.form_submit_button("Confirm Mapping")

        # If user confirms the mapping
        if submit_mapping:
            st.session_state.activity_mapping = activity_mapping
            st.session_state.mapping_complete = True
            st.success("Mapping confirmed.")


    # SECTION 4: RUN COMPARE_TIME FUNCTION ------------------------------------------------------------------------------------------------------
    if st.session_state.mapping_complete:
        if st.button("Run Comparison"):
            compare_time(legacy_df, wfs_df, threshold, st.session_state.activity_mapping, column_pairs, num_records_legacy)
            st.success("Comparison completed!")

    # SECTION 5: DISPLAY RESULTS ----------------------------------------------------------------------------------------------------------------
