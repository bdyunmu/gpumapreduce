/*
	Copyright 2012 The Trustees of Indiana University. All rights reserved.
	Panda:a MapReduce Framework on GPUs and CPUs
	Time: 2018-5-9
	Developer: Hui Li (huili@ruijie.com.cn)
*/

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <stdio.h>

  class TeraSort {
  //implicit val caseInsensitiveOrdering : Comparator[Array[Byte]] =
  //  UnsignedBytes.lexicographicalComparator
  	int Comparator(){
	return 0;
	}
  };

#if 0
  int main(int argc, char **argv){

  //def main(args: Array[String]) {
  //  if (args.length < 2) {
  //    println("Usage:")
  /*    println("DRIVER_MEMORY=[mem] spark-submit " +
        "com.github.ehiggs.spark.terasort.TeraSort " +
        "spark-terasort-1.0-SNAPSHOT-with-dependencies.jar " +
        "[input-file] [output-file]")
      println(" ")
      println("Example:")
      println("DRIVER_MEMORY=50g spark-submit " +
        "com.github.ehiggs.spark.terasort.TeraSort " +
        "spark-terasort-1.0-SNAPSHOT-with-dependencies.jar " +
        "/home/myuser/terasort_in /home/myuser/terasort_out")
      System.exit(0)
   */
	if(argc<3){
	printf("Usage:\n");
	printf("%s [input-file][output-file]\n",argv[0]);
	exit(-1);
	}//if
	return 0;

    }//int main
#endif

    // Process command line arguments
    //val inputFile = args(0)
    //val outputFile = args(1)

    //val conf = new SparkConf()
    //  .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //  .setAppName(s"TeraSort")
    //val sc = new SparkContext(conf)

    //val dataset = sc.newAPIHadoopFile[Array[Byte], Array[Byte], TeraInputFormat](inputFile)
    //val sorted = dataset.repartitionAndSortWithinPartitions(
    //  new TeraSortPartitioner(dataset.partitions.length))
    //sorted.saveAsNewAPIHadoopFile[TeraOutputFormat](outputFile)
    //System.exit(0) //explicitly exiting
