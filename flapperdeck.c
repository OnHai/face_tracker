#define DEBUG_MODULE "FLAPPER"

#include "FreeRTOS.h"
#include "task.h"

#include "deck.h"
#include "system.h"
#include "debug.h"
#include "log.h"
#include "param.h"
#include "extrx.h"
#include "flapperdeck.h"
#include "pm.h"
#include "autoconf.h"
#include "config.h"
#include "uart2.h"

// static float reading_last = 0.0;
// static float current_last = 0.0;
static float current = 0.0;
static float vbat = 0.0;
static float power = 0.0;

static float ampsPerVolt = 2.5; 
static float filter_alpha = 0.975;

static float rpi_val = 0.0;
static float rpi_y = 0.0;
static float rpi_face_size = 0.0;

static bool isInit;

/* bin 2 float */
typedef union {
  float value;
  uint8_t bytes[4];
} FloatBuffer_t;

void flapperDeckInit(DeckInfo* info)
{
  if (isInit)
    return;


  uart2Init(115200);

  xTaskCreate(flapperDeckTask, FLAPPERDECK_TASK_NAME, FLAPPERDECK_TASK_STACKSIZE, NULL, FLAPPERDECK_TASK_PRI, NULL);

  #if CONFIG_DECK_FLAPPER_EXTRX_ENABLE
  extRxInit();
  #endif

  isInit = true;
}

bool flapperDeckTest(void)
{
  if (!isInit)
    return false;

  return uart2Test();
}

void flapperDeckTask(void* arg)
{
  systemWaitStart();
  TickType_t xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  uint8_t header;
  FloatBuffer_t f1, f2, f3;

while (1) {
    // polling
    if (uart2GetCharWithTimeout(&header, 1)) {
        if (header == 'S') {
            if (uart2GetDataWithTimeout(4, f1.bytes, M2T(5)) == 4 &&
                uart2GetDataWithTimeout(4, f2.bytes, M2T(5)) == 4 &&
                uart2GetDataWithTimeout(4, f3.bytes, M2T(5)) == 4) {
                rpi_val = f1.value;  // angle_x
                rpi_y = f2.value;  // angle_y
                rpi_face_size = f3.value;  // diagonal 
            }
        }
    }

    vTaskDelayUntil(&xLastWakeTime, M2T(1));
}
}

static const DeckDriver flapper_deck = {
  .vid = 0xBC,
  .pid = 0x09,
  .name = "bcFlapperDeck",
  #if CONFIG_DECK_FLAPPER_EXTRX_ENABLE
  .usedPeriph = DECK_USING_UART2 | DECK_USING_TIMER9,
  #else
  .usedPeriph = DECK_USING_UART2,
  #endif
  .init = flapperDeckInit,
  .test = flapperDeckTest,
};

DECK_DRIVER(flapper_deck);

PARAM_GROUP_START(deck)
PARAM_ADD_CORE(PARAM_UINT8 | PARAM_RONLY, bcFlapperDeck, &isInit)
PARAM_GROUP_STOP(deck)

LOG_GROUP_START(flapper)
LOG_ADD(LOG_FLOAT, vbat, &vbat)
//LOG_ADD(LOG_FLOAT, i_raw, &current_last)
LOG_ADD(LOG_FLOAT, current, &current)
LOG_ADD(LOG_FLOAT, power, &power)
LOG_ADD(LOG_FLOAT, rpi_val, &rpi_val)
LOG_ADD(LOG_FLOAT, rpi_y, &rpi_y)    
LOG_ADD(LOG_FLOAT, rpi_face_size, &rpi_face_size) 
LOG_GROUP_STOP(flapper)

PARAM_GROUP_START(flapper)
PARAM_ADD(PARAM_FLOAT | PARAM_PERSISTENT, ampsPerVolt, &ampsPerVolt)
PARAM_ADD(PARAM_FLOAT | PARAM_PERSISTENT, filtAlpha, &filter_alpha)
PARAM_GROUP_STOP(flapper)