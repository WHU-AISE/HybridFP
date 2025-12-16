package org.apache.openwhisk.core.containerpool

import java.time.Instant


//
// Record wasted time of memory allocated to containers
//

class WastedMemoryTimeRecorder() {

  var bareWastedMemoryTime: Double = 0.0
  var start: Instant = Instant.EPOCH
  var end: Instant = Instant.EPOCH


  var bareTimeline: String = ""

  def setStart(t: Instant) = {
    start = t
  }

  def setEnd(t: Instant) = {
    end = t
  }

  def summary(memory: Double, status: String) = {
    if (start != Instant.EPOCH && end != Instant.EPOCH) {
      val interval = Interval(start, end)
      val wastedMemory: Double = memory / 1024 / 1024 // byte to Mb
      bareWastedMemoryTime = bareWastedMemoryTime + wastedMemory * (interval.duration.length.toDouble / 1000) // megabyte x second
      println(s" ${wastedMemory},${start.toEpochMilli},${end.toEpochMilli},${status}")
      if (bareTimeline == "") {
        bareTimeline = s"${wastedMemory},${start.toEpochMilli},${end.toEpochMilli},${status}"
      } else {
        bareTimeline = bareTimeline + s" ${wastedMemory},${start.toEpochMilli},${end.toEpochMilli},${status}"
      }
    }
    start = Instant.EPOCH
    end = Instant.EPOCH
  }

  def timeline: Option[String] = Some(s"${bareTimeline}")

  def total: Double = {bareWastedMemoryTime}
}

