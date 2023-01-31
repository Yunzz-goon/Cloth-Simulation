/* 
 * File:   simple_papi.h
 * Author: jaros
 *
 * Created on 14 May 2012, 12:15 PM
 */

#ifndef SIMPLE_PAPI_H
#define	SIMPLE_PAPI_H



// Init PAPI
 void InitPapi();

// Start PAPI counters and return time 
long long PapiStartCounters();

// Stop PAPI counters
long long PapiStopCounters();


// Print PAPI results
void PrintPapiResults( const char * RoutineName, 
                       long long StartTime, long long StopTime);






#endif	/* SIMPLE_PAPI_H */

