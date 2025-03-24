

from crosscoders.dataclasses.runner import JOB_TYPE_ENUM
from crosscoders.runners.data import DataRunner




class SparkDataRunner(DataRunner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        if self.cfg['job'] == JOB_TYPE_ENUM.glue:

            import sys

            from awsglue.utils import getResolvedOptions
            from pyspark.context import SparkContext
            from awsglue.context import GlueContext
            # from awsglue.job import Job

            args = getResolvedOptions(sys.argv, ['JOB_NAME'])

            sc = SparkContext()
            glueContext = GlueContext(sc)
            self.spark = glueContext.spark_session

            # job = Job(glueContext)
            # job.init(args['JOB_NAME'], args)
            # job.commit()

        else:   # TODO
            ...


class ShuffleDataRunner(SparkDataRunner):

    def run(self):

        ds_bucket = 'crosscoders'
        ds_prefix = 'input/roneneldan/TinyStories/train/tiny-stories-33M'
        ds_subset_name = '100M'

        # input_path = 's3://crosscoders/input/roneneldan/TinyStories/train/tiny-stories-33M/100M/'
        input_path = f's3://{ds_bucket}/{ds_prefix}/{ds_subset_name}/'

        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType
        schema = StructType([
            StructField('tokens', ArrayType(IntegerType())),
            StructField('tiny-stories-33M.ln2.normalized', ArrayType(FloatType())),
            StructField('tiny-stories-33M.mlp_out', ArrayType(FloatType())),
            StructField('tiny-stories-33M.resid_post', ArrayType(FloatType())),

        ])
        df = self.spark.read.parquet(input_path, schema=schema)


        import pyspark.sql.functions as F

        # df = df.orderBy(F.rand(seed=314159))
        num_partitions = 10000
        df = df.repartition(num_partitions, F.rand(seed=314159))

        df.printSchema()


        output_path = f's3://{ds_bucket}/{ds_prefix}/{ds_subset_name}-shuffled/'

        df.write.parquet(output_path, mode='overwrite')


class StatisticsDataRunner(SparkDataRunner):

    def run(self):

        ds_bucket = 'crosscoders'
        ds_prefix = 'input/roneneldan/TinyStories/train/tiny-stories-33M'
        ds_subset_name = '100M'

        # input_path = 's3://crosscoders/input/roneneldan/TinyStories/train/tiny-stories-33M/100M/'
        input_path = f's3://{ds_bucket}/{ds_prefix}/{ds_subset_name}/'

        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType
        schema = StructType([
            StructField('tokens', ArrayType(IntegerType())),
            StructField('tiny-stories-33M.ln2.normalized', ArrayType(FloatType())),
            StructField('tiny-stories-33M.mlp_out', ArrayType(FloatType())),
            StructField('tiny-stories-33M.resid_post', ArrayType(FloatType())),

        ])
        df = self.spark.read.parquet(input_path, schema=schema)
        df.printSchema()


        import pyspark.sql.functions as F
        from pyspark.sql.functions import array, arrays_zip, explode, col, mean, lit, pandas_udf
        from pyspark.sql.types import ArrayType, FloatType, IntegerType, StructType, StructField
        import numpy as np
        import pandas as pd

        # Define dimensions
        rows = 4
        cols = 768

        # Use pandas_udf (vectorized UDF) instead of regular UDF for better performance
        @pandas_udf(ArrayType(FloatType()))
        def reshape_and_l2_norm_pandas(series):
            def process(flattened_array):
                if flattened_array is None or len(flattened_array) == 0:
                    return None
                # Use float32 explicitly to match your data type
                reshaped = np.array(flattened_array, dtype=np.float32).reshape(rows, cols)
                # L2 norm along axis 1
                l2_norms = np.linalg.norm(reshaped, axis=1).astype(np.float32)
                return l2_norms.tolist()

            return series.apply(process)

        # Create a single UDF for positions
        @pandas_udf(ArrayType(IntegerType()))
        def layer_idx_udf(*args):
            return pd.Series([[0, 1, 2, 3]])

        @pandas_udf(ArrayType(IntegerType()))
        def layer_idx_udf(*args):
            return pd.Series([[1, 2, 3, 4]] * len(args[0]))

        # Main processing pipeline
        # 1. Read the data with the right schema and partitioning
        # You might want to adjust partition count based on your cluster
        # df = spark.read.parquet("your_data_path").repartition(1000)
        # acts_df = df.limit(10)


        columns = [c for c in df.columns if c != 'tokens']

        # 2. Apply the UDFs to calculate L2 norms
        # Process the arrays - use persisting to avoid recomputation
        # processed_df = df \
        #     .select(
        #         # positions_pandas(lit(1)).alias('layer_idx'),
        #         array(lit(1), lit(2), lit(3), lit(4)).alias('layer_idx')
        #         *[reshape_and_l2_norm_pandas(f"`{c}`").alias(c) for c in columns]
        #     )
        processed_df = df.select(
            # F.array(F.lit(1), F.lit(2), F.lit(3), F.lit(4)).alias('layer_idx'),
            layer_idx_udf(lit(1)).alias('layer_idx'),
            *[reshape_and_l2_norm_pandas(f"`{c}`").alias(c) for c in columns]
        )

        columns = ['layer_idx'] + columns

        # 3. Optimize by selecting only necessary columns
        slim_df = processed_df #.select(*columns)
        slim_df.printSchema()

        # slim_df.show(50)


        slim_df.persist()  # Persist for multiple operations

        # 4. Zip and explode in one operation
        exploded_df = slim_df \
            .select(explode(arrays_zip(*[f'`{c}`' for c in columns])).alias("norm_tuple")) \
            .select(*[col('norm_tuple')[c].alias(c) for c in columns])

        # 5. Repartition by position for more efficient grouping
        # With 4 positions, this creates 4 partitions, one for each position
        layers_df = exploded_df  #.repartition(col("position"))
        layers_df.printSchema()

        layers_df.persist()



        # 6. Compute the means at position level
        layer_means = layers_df.groupBy("layer_idx").agg(
            *[mean(f'`{c}`').alias(c) for c in columns]
        )

        # 7. Compute overall means efficiently
        # Since we have just 4 positions and 3 arrays, this approx. 12 values aggregation is tiny
        overall_means = layers_df.agg(
            *[mean(f'`{c}`').alias(c) for c in columns if c != 'layer_idx']
        )

        # 8. Collect the overall means as local variables (very small data)
        overall_means = overall_means.collect()[0].asDict()


        # 9. Attach overall means to position means without a join
        # final_result = position_means.withColumn(
        #     "overall_mean_l2_norm1", lit(overall_values["overall_mean_l2_norm1"])
        # ).withColumn(
        #     "overall_mean_l2_norm2", lit(overall_values["overall_mean_l2_norm2"])
        # ).withColumn(
        #     "overall_mean_l2_norm3", lit(overall_values["overall_mean_l2_norm3"])
        # )

        layer_means = layer_means.collect()
        layer_means = {row.layer_idx: row.asDict() for row in layer_means}

        print(layer_means)
        print(overall_means)

        # Clean up
        # slim_df.unpersist()


        def save_stats(bucket, prefix, stats):
            import boto3
            import json

            s3object = boto3.resource('s3') \
                .Object(bucket, prefix)

            s3object.put(Body=(bytes(json.dumps(stats).encode('UTF-8'))))



        ds_subset_name = '100M-overall'
        stats_path = f'{ds_prefix}/stats/{ds_subset_name}.json'
        save_stats(ds_bucket, stats_path, overall_means)


        ds_subset_name = '100M-layers'
        stats_path = f'{ds_prefix}/stats/{ds_subset_name}.json'
        save_stats(ds_bucket, stats_path, layer_means)




