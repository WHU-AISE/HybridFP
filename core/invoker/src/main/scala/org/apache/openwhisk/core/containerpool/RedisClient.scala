package org.apache.openwhisk.core.containerpool

import java.time.Instant
import org.apache.openwhisk.common._
import org.apache.commons.pool2.impl.GenericObjectPoolConfig
import redis.clients.jedis.JedisPool
import spray.json._
import DefaultJsonProtocol._
import scala.collection.JavaConverters._

class RedisClient(
                   host: String = "172.17.0.1",
                   port: Int = 6379,
                   password: String = "openwhisk",
                   database: Int = 0,
                   logging: Logging
                 ) {
  private var pool: JedisPool = _
  val interval: Int = 1000 //ms

  def init: Unit = {
    val maxTotal: Int = 300
    val maxIdle: Int = 100
    val minIdle: Int = 1
    val timeout: Int = 30000

    val poolConfig = new GenericObjectPoolConfig()
    poolConfig.setMaxTotal(maxTotal)
    poolConfig.setMaxIdle(maxIdle)
    poolConfig.setMinIdle(minIdle)
    pool = new JedisPool(poolConfig, host, port, timeout, password, database)
  }

  def getPool: JedisPool = {
    assert(pool != null)
    pool
  }

  //
  // Send observations to Redis
  //
  def setWastedMemoryTime(containerId: String, wastedMemoryTimeLine: String, wastedMemoryTime: Double): Unit = {
    try {
      val jedis = pool.getResource
      val name: String = "wasted_memory_timeline"
      val key: String = containerId
      val value: String = wastedMemoryTimeLine
      jedis.hset(name, key, value)
      println(s"current total wasted memorytime: ${wastedMemoryTime} , set ${containerId} WastedMemoryTime ${wastedMemoryTimeLine}")
      jedis.set("total_wasted_memorytime", f"$wastedMemoryTime%.2f")
      jedis.close()
    } catch {
      case e: Exception => {
        logging.error(this, s"set wasted memory time error, exception ${e}, at ${Instant.now.toEpochMilli}")
      }
    }
  }

  /**
   * Get full prediction arrays from Redis for container prewarming.
   * Each array contains 60 integers representing predicted invocations for next 60 minutes.
   * 
   * @return Map of function ID (namespace/action) to array of 60 prediction values
   */
  def getPredictions(): Map[String, Array[Int]] = {
    try {
      val jedis = pool.getResource
      val predictions = jedis.hgetAll("prediction")
      println(s"predictions: $predictions")
      jedis.close()
      
      if (predictions == null || predictions.isEmpty) {
        logging.info(this, "No predictions found in Redis")
        return Map.empty[String, Array[Int]]
      }
      
      // Parse JSON array - should contain 60 integers
      val result = predictions.asScala.flatMap { case (funcId, jsonArray) =>
        try {
          val arr = jsonArray.parseJson.convertTo[List[Int]].toArray
          println(s"arr: $arr")
          if (arr.length == 60) {
            Some(funcId -> arr)
          } else {
            logging.warn(this, s"Prediction array for $funcId has ${arr.length} elements, expected 60")
            None
          }
        } catch {
          case e: Exception =>
            logging.warn(this, s"Failed to parse prediction for $funcId: ${e.getMessage}")
            None
        }
      }.toMap
      
      if (result.nonEmpty) {
        logging.info(this, s"Retrieved ${result.size} prediction arrays from Redis (60 minutes each)")
      }
      result
      
    } catch {
      case e: Exception =>
        logging.error(this, s"Failed to get predictions from Redis: ${e.getMessage}")
        Map.empty[String, Array[Int]]
    }
  }

}