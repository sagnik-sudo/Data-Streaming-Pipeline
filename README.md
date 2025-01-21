# Data-Streaming-Pipeline

## Project Plans and Goals

1. **Ingestion of Data from Parquet**
   - The initial stage involves ingesting data from Parquet files. This data serves as the raw input for the pipeline.

2. **Cleaning of Data using Apache Spark**
   - The ingested data is cleaned and transformed using Apache Spark. This step ensures that the data is in the correct format and free of any inconsistencies or errors.

3. **Batch Streaming of Data using Apache Kafka**
   - The cleaned data is then streamed in batches using Apache Kafka. Kafka acts as a distributed streaming platform that handles the real-time data flow.

4. **Ingestion from Receiver, and Storage into Real-Time Table**
   - The data streamed via Kafka is ingested by a receiver and stored into a real-time table. This table is designed to handle high-velocity data and provide quick access.

5. **Data Processing and Visualization**
   - The data is then passed through a view and displayed on a dashboard. The same data is also sent to a Data Sink for archival and long-term storage.

6. **Dashboard Features**
   - The dashboard not only displays incoming data details but also provides insights into archived data. This dual functionality ensures that users can monitor real-time data and access historical data for analysis.

By following these stages, the project aims to create a robust data streaming pipeline that efficiently handles data ingestion, cleaning, streaming, storage, and visualization.


## Setup and Running the Project

### Prerequisites
- Ensure you have the following software installed:
  - Apache Spark
  - Apache Kafka
  - Java Development Kit (JDK)
  - Python (with necessary libraries)
  - Docker (optional, for containerized deployment)

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/data-streaming-pipeline.git
   cd data-streaming-pipeline
   ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```


