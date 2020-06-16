
#include "main.h"
#include "stm32f4xx_it.h"

extern DMA_HandleTypeDef hdma_adc1;
extern ADC_HandleTypeDef hadc1;

void NMI_Handler(void)
{

}

void HardFault_Handler(void)
{

  while (1)
  {

  }
}


void MemManage_Handler(void)
{

  while (1)
  {

  }
}


void BusFault_Handler(void)
{

  while (1)
  {

  }
}


void UsageFault_Handler(void)
{

  while (1)
  {

  }
}


void SVC_Handler(void)
{

}


void DebugMon_Handler(void)
{

}


void PendSV_Handler(void)
{

}


void SysTick_Handler(void)
{

  HAL_IncTick();

}

void EXTI4_IRQHandler(void)
{

  HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_4);

		for(int i=0;i<2000;i++);

	//HAL_ADC_Start_DMA(&hadc1,(uint32_t *) sdata,8000);

}

void ADC_IRQHandler(void)
{

  HAL_ADC_IRQHandler(&hadc1);

}

void DMA2_Stream0_IRQHandler(void)
{

  HAL_DMA_IRQHandler(&hdma_adc1);

}

