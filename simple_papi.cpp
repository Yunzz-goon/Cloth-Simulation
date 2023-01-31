#include <papi.h>
#include <stdio.h>
#include <cstdlib>


static const int NUM_EVENTS = 4;
static int       EVENTS[NUM_EVENTS] = {PAPI_DP_OPS, PAPI_TOT_CYC, PAPI_LST_INS, PAPI_L1_DCM};
static long long VALUES[NUM_EVENTS];



/*
 * Init PAPI
 */
void InitPapi(){
   /* Start Init library */
   if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT )
     {
     fprintf(stderr,"PAPI Library initialization error! %d\n",  __LINE__);
     exit(1);
   }    
  
}// end InitPapi
//-----------------------------------------------------------------------------


/*
 * Start papi counters and return time
 */
long long PapiStartCounters(){

    int retval = 0;
    
/* Start counting events */
   if ((retval = PAPI_start_counters(EVENTS, NUM_EVENTS)) != PAPI_OK)
   {
    fprintf(stderr,"PAPI Start counter error! %d, %d\n",  retval, __LINE__);
    exit(1);
   }    

   return(PAPI_get_real_usec());  
}// end of Papi_Start_Counters
//------------------------------------------------------------------------------

/*
 * Stop Papi counters
 */
long long PapiStopCounters(){

   long long StopTime = PAPI_get_real_usec();  
   int retval = 0;
   
    /* Stop counting events */
    if ((retval = PAPI_stop_counters(VALUES, NUM_EVENTS)) != PAPI_OK){    
       fprintf(stderr,"PAPI stop counters error! %d, %d\n", retval, __LINE__);
       exit(1);
    }

   return (StopTime);
}// end of PapiStopCounters
//------------------------------------------------------------------------------

/*
 * Print Papi results
 */
void PrintPapiResults( const char * RoutineName, 
                       long long StartTime, long long StopTime){
    
   //  printf("_____Routine: %s_______\n", RoutineName); 
    printf("Exec. time (ms): %20.3f\n", (StopTime - StartTime)/ (double)1000);
   //  printf("PAPI_MFLOP_OPS:     %20lld\n", VALUES[0]/1000000);
   //  printf("PAPI_TOT_CYC:    %20lld\n", VALUES[1]);
   //  printf("MFLOPS:          %20.3f\n",  (double) VALUES[0] / (double) (StopTime - StartTime));
   //  printf("FP per cycle:    %20.3f\n",  (double) VALUES[0] / (double) VALUES[1]);
   //  printf("Cycles per sec:  %20lld\n", VALUES[1]/ (StopTime - StartTime));
   //  printf("Load/Write time:  %20lld\n", VALUES[2]);
   //  printf("L1 miss number:  %20lld\n", VALUES[3]);

    
}// end of PrintPapiResults
//------------------------------------------------------------------------------



