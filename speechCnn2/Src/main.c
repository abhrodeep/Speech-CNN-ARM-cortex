#include "main.h"
#include "stm32f4xx_hal.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_const_structs.h"
#include "weights_CL1.h"
#include "weights_CL2.h"
#include "weights_CL3.h"
#include "weights_FC1.h"



ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;


void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_ADC1_Init(void);

//CL1
q7_t CONV1_OUT[CONV1_OUT_CH*CONV1_OUT_x*CONV1_OUT_y];
//int8_t conv1[CONV1_OUT_CH][CONV1_OUT_y][CONV1_OUT_x];
q7_t CONV1_MAX_OUT[MAX1_CHANNEL_in*MAX1_OUT_y*MAX1_OUT_x];

//CL2
q7_t CONV2_OUT[CONV2_OUT_CH*CONV2_OUT_x*CONV2_OUT_y];
q7_t CONV2_MAX_OUT[MAX2_CHANNEL_in*MAX2_OUT_y*MAX2_OUT_x];

//CL3
q7_t CONV3_OUT[CONV3_OUT_CH*CONV3_OUT_x*CONV3_OUT_y];
q7_t CONV3_MAX_OUT[MAX3_CHANNEL_in*MAX3_OUT_y*MAX3_OUT_x];


//FC1
q7_t FC1_OUT[IP1_OUT_DIM];

//Buffers			
q7_t col_buffer1[10000];

q15_t buffer3[IP1_IN_DIM];
float32_t detrend=0;
uint8_t sdata[16000];  //raw 8-bit sample from ADC 
float32_t XTest[8000];	//buffer after conversion for each half
int i=0,j=0,m=0,count=0,k=0,p=0,h=0,loc;
int n=sizeof(XTest)/sizeof(float);
float32_t x[256],addf=0;
float32_t spec[39][128];
float32_t specP[1248];
float32_t specPT[1248];
float32_t testOutput2[256];
float32_t testMag2[128];
uint32_t fftSize = 256;
uint32_t ifftFlag = 0;
arm_rfft_fast_instance_f32 S2;
arm_status status1;
int hop=200;
int windowsize=240;
int8_t data[1248], max_fout;
arm_matrix_instance_f32 A;      /* Matrix A Instance */
arm_matrix_instance_f32 AT;     /* Matrix AT(A transpose) instance */


void arm_max(const uint16_t 	input_y,
const uint16_t 	input_x,
const uint16_t 	output_y,
const uint16_t 	output_x,
const uint16_t 	stride_y,
const uint16_t 	stride_x,
const uint16_t 	kernel_y,
const uint16_t 	kernel_x,
const uint16_t 	pad_y,
const uint16_t 	pad_x,
const int8_t 	act_min,
const int8_t 	act_max,
const uint16_t 	channel_in,
int8_t * 	input,
int16_t * 	tmp_buffer,
int8_t * 	output 
)
{
	
	int32_t i_ch_in, i_out_x, i_out_y;
    int32_t i_ker_x, i_ker_y,ccc;
    (void)tmp_buffer;
 int32_t ker_y_start;
 int32_t ker_x_start;
 int32_t ker_y_end;
 int32_t ker_x_end;
	
    for (i_out_y = 0; i_out_y < output_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < output_x; i_out_x++)
        {
            for (i_ch_in = 0; i_ch_in < channel_in; i_ch_in++)
            {
                /* Native data type for inner loop variables  */
                int32_t max_val = -128;
                /* Condition for kernel start dimension: (base_idx_<x,y> + ker_<x,y>_start) >= 0 */
                const int32_t base_idx_y = (i_out_y * stride_y) - pad_y;
                const int32_t base_idx_x = (i_out_x * stride_x) - pad_x;
							
                
if((-base_idx_y)>0)
							{
								ker_y_start=(-base_idx_y);
							}
							else
							{
								ker_y_start=0;
							}
							if((-base_idx_x)>0)
							{
								ker_x_start=(-base_idx_x);
															
							}
							else
							{
								ker_x_start=0;
							}
                /* Condition for kernel end dimension: (base_idx_<x,y> + ker_<x,y>_end) < input_<x,y> */
               
if((input_y - base_idx_y)<kernel_y)
							{
								ker_y_end=input_y - base_idx_y;
							}
							else
							{
								ker_y_end=kernel_y;
							}
							if((input_x - base_idx_x)<kernel_x)
							{
								ker_x_end=(input_x - base_idx_x);
							}
							else
							{
								ker_x_end=kernel_x;
							}
												
							
                for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                {
                    for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                    {
                        const int32_t col_idx = base_idx_x + i_ker_x;
                        const int32_t row_idx = base_idx_y + i_ker_y;
                        ccc=input[(row_idx * input_x + col_idx) * channel_in + i_ch_in];
											if(max_val<= ccc)
															max_val = ccc;
                    }
                }

                /* Activation function */
                //max_val = MAX(max_val, act_min);
                //max_val = MIN(max_val, act_max);

                output[i_ch_in + channel_in * (i_out_x + i_out_y * output_x)] = (int8_t)max_val;
								//output[(i_ch_in*output_y*output_x) + (i_out_y*output_x)+i_out_x] = (int8_t)max_val;
            }
        }
    }
   
	
}

// for fourier transform and spectrum generation
void buffer_cr(int n,int windowsize,int hop)
{
 for(i=0;i<n-hop;i=i+hop)
	{
  
  for(k=0,j=i; k<256 && j<(i+256);k++,j++) //&&(n-count*hop))
  {
   x[k]=XTest[j];
		if(k>windowsize-1)
			x[k]=0;
  }
			
	k=0;   //break_point
	status1= arm_rfft_fast_init_f32	(&S2,256);
	arm_rfft_fast_f32	(&S2,x,testOutput2,ifftFlag);
	spec[count][127]=fabs(testOutput2[1]);
  arm_cmplx_mag_f32(testOutput2, testMag2, fftSize);
	
	for(m=0;m<128;m++){
		if(m!=127){
			spec[count][m]=testMag2[m+1];} //xx=Y([2:129],:)
		
			addf=addf+spec[count][m];
			
		if(p==3){
			//P[count][((m+1)/4)-1]=(addf/4.0);		//P(ii+1,:)=sum(xx(((ii*4)+1):((ii+1)*4),:))/4;
			specP[h]=log10f((addf/4.0));
			//data[h]=log10f((addf/4.0));
			addf=0;
			p=0;
			h++;
		}
		else{
		p++;}
		}
	
	count++;
  }
	
	arm_mat_init_f32(&A, 39, 32, (float32_t *)specP);
	arm_mat_init_f32(&AT, 32, 39, (float32_t *)specPT);
	arm_mat_trans_f32(&A, &AT);
	h=0;
	count=0;
	p=0;
 }


int main(void)
{

  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();

	HAL_ADC_Start_DMA(&hadc1,(uint32_t *) sdata,16000);

  while (1)
  {
    
  }

}

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef* hadc)
{
  
  UNUSED(hadc);
  
	detrend=0;
	for(int i=0;i<8000;i++){
	detrend+=sdata[i];
	}
	detrend=detrend/8000.0;
	
	for(int i=0;i<8000;i++){
	XTest[i]=sdata[i]-detrend;
	}
	
	buffer_cr(n,windowsize,hop);
//	
	for(i=0;i<1248;i++){
		data[i]=round(16*(specPT[i]-mean[i]));
	}
	
  arm_convolve_HWC_q7_basic_nonsquare(data,CONV1_IN_x,CONV1_IN_y,CONV1_IN_CH,W_1,CONV1_OUT_CH,CONV1_KER_x,CONV1_KER_y,CONV1_PAD_x,CONV1_PAD_y,CONV1_STRIDE_y,
               CONV1_STRIDE_x,b_1,CONV1_BIAS_LSHIFT,CONV1_OUT_RSHIFT,CONV1_OUT,CONV1_OUT_x,CONV1_OUT_y,(q15_t*)col_buffer1,NULL);
arm_relu_q7(CONV1_OUT,CONV1_OUT_x*CONV1_OUT_y*CONV1_OUT_CH);
arm_max(MAX1_IN_y,MAX1_IN_x,MAX1_OUT_y,MAX1_OUT_x,MAX1_STRIDE_y,MAX1_STRIDE_x,MAX1_KERNEL_y,MAX1_KERNEL_x,MAX1_PAD_y,MAX1_PAD_x,MAX1_ACT_min,MAX1_ACT_max,MAX1_CHANNEL_in,CONV1_OUT,NULL,CONV1_MAX_OUT);

	//CL2	

	arm_convolve_HWC_q7_fast_nonsquare(CONV1_MAX_OUT,CONV2_IN_x,CONV2_IN_y,CONV2_IN_CH,W_2,CONV2_OUT_CH,CONV2_KER_x,CONV2_KER_y,CONV2_PAD_x,CONV2_PAD_y,CONV2_STRIDE_y,
               CONV2_STRIDE_x,b_2,CONV2_BIAS_LSHIFT,CONV2_OUT_RSHIFT,CONV2_OUT,CONV2_OUT_x,CONV2_OUT_y,(q15_t*)col_buffer1,NULL);
arm_relu_q7(CONV2_OUT,CONV2_OUT_x*CONV2_OUT_y*CONV2_OUT_CH);
arm_max(MAX2_IN_y,MAX2_IN_x,MAX2_OUT_y,MAX2_OUT_x,MAX2_STRIDE_y,MAX2_STRIDE_x,MAX2_KERNEL_y,MAX2_KERNEL_x,MAX2_PAD_y,MAX2_PAD_x,MAX2_ACT_min,MAX2_ACT_max,MAX2_CHANNEL_in,CONV2_OUT,NULL,CONV2_MAX_OUT);

	//CL3	
	arm_convolve_HWC_q7_fast_nonsquare(CONV2_MAX_OUT,CONV3_IN_x,CONV3_IN_y,CONV3_IN_CH,W_3,CONV3_OUT_CH,CONV3_KER_x,CONV3_KER_y,CONV3_PAD_x,CONV3_PAD_y,CONV3_STRIDE_y,
               CONV3_STRIDE_x,b_3,CONV3_BIAS_LSHIFT,CONV3_OUT_RSHIFT,CONV3_OUT,CONV3_OUT_x,CONV3_OUT_y,(q15_t*)col_buffer1,NULL);
arm_relu_q7(CONV3_OUT,CONV3_OUT_x*CONV3_OUT_y*CONV3_OUT_CH);
arm_max(MAX3_IN_y,MAX3_IN_x,MAX3_OUT_y,MAX3_OUT_x,MAX3_STRIDE_y,MAX3_STRIDE_x,MAX3_KERNEL_y,MAX3_KERNEL_x,MAX3_PAD_y,MAX3_PAD_x,MAX3_ACT_min,MAX3_ACT_max,MAX3_CHANNEL_in,CONV3_OUT,NULL,CONV3_MAX_OUT);
	//FC1				
	
	arm_fully_connected_q7(CONV3_MAX_OUT,wf_1,IP1_IN_DIM,IP1_OUT_DIM,IP1_BIAS_LSHIFT,IP1_OUT_RSHIFT,bf_1,FC1_OUT,buffer3);	

		max_fout=FC1_OUT[0];
	loc=0;
	
	for(int i=0; i<12;i++){
	if(FC1_OUT[i]>max_fout)
	{
		max_fout=FC1_OUT[i];
		loc=i;
	}
	}
	GPIOD->ODR=(val[loc]<<9);

}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
{
 
  UNUSED(hadc);
	
	detrend=0;
	for(int i=8000;i<16000;i++){
	detrend+=sdata[i];
	}
	detrend=detrend/8000.0;
	
	for(int i=0;i<8000;i++){
	XTest[i]=sdata[8000+i]-detrend;
	}
	
	buffer_cr(n,windowsize,hop);
	
	for(i=0;i<1248;i++){
		data[i]=round(16*(specPT[i]-mean[i]));
	}
	
  arm_convolve_HWC_q7_basic_nonsquare(data,CONV1_IN_x,CONV1_IN_y,CONV1_IN_CH,W_1,CONV1_OUT_CH,CONV1_KER_x,CONV1_KER_y,CONV1_PAD_x,CONV1_PAD_y,CONV1_STRIDE_y,
               CONV1_STRIDE_x,b_1,CONV1_BIAS_LSHIFT,CONV1_OUT_RSHIFT,CONV1_OUT,CONV1_OUT_x,CONV1_OUT_y,(q15_t*)col_buffer1,NULL);
arm_relu_q7(CONV1_OUT,CONV1_OUT_x*CONV1_OUT_y*CONV1_OUT_CH);
arm_max(MAX1_IN_y,MAX1_IN_x,MAX1_OUT_y,MAX1_OUT_x,MAX1_STRIDE_y,MAX1_STRIDE_x,MAX1_KERNEL_y,MAX1_KERNEL_x,MAX1_PAD_y,MAX1_PAD_x,MAX1_ACT_min,MAX1_ACT_max,MAX1_CHANNEL_in,CONV1_OUT,NULL,CONV1_MAX_OUT);

	//CL2	

	arm_convolve_HWC_q7_fast_nonsquare(CONV1_MAX_OUT,CONV2_IN_x,CONV2_IN_y,CONV2_IN_CH,W_2,CONV2_OUT_CH,CONV2_KER_x,CONV2_KER_y,CONV2_PAD_x,CONV2_PAD_y,CONV2_STRIDE_y,
               CONV2_STRIDE_x,b_2,CONV2_BIAS_LSHIFT,CONV2_OUT_RSHIFT,CONV2_OUT,CONV2_OUT_x,CONV2_OUT_y,(q15_t*)col_buffer1,NULL);
arm_relu_q7(CONV2_OUT,CONV2_OUT_x*CONV2_OUT_y*CONV2_OUT_CH);
arm_max(MAX2_IN_y,MAX2_IN_x,MAX2_OUT_y,MAX2_OUT_x,MAX2_STRIDE_y,MAX2_STRIDE_x,MAX2_KERNEL_y,MAX2_KERNEL_x,MAX2_PAD_y,MAX2_PAD_x,MAX2_ACT_min,MAX2_ACT_max,MAX2_CHANNEL_in,CONV2_OUT,NULL,CONV2_MAX_OUT);

	//CL3	
	arm_convolve_HWC_q7_fast_nonsquare(CONV2_MAX_OUT,CONV3_IN_x,CONV3_IN_y,CONV3_IN_CH,W_3,CONV3_OUT_CH,CONV3_KER_x,CONV3_KER_y,CONV3_PAD_x,CONV3_PAD_y,CONV3_STRIDE_y,
               CONV3_STRIDE_x,b_3,CONV3_BIAS_LSHIFT,CONV3_OUT_RSHIFT,CONV3_OUT,CONV3_OUT_x,CONV3_OUT_y,(q15_t*)col_buffer1,NULL);
arm_relu_q7(CONV3_OUT,CONV3_OUT_x*CONV3_OUT_y*CONV3_OUT_CH);
arm_max(MAX3_IN_y,MAX3_IN_x,MAX3_OUT_y,MAX3_OUT_x,MAX3_STRIDE_y,MAX3_STRIDE_x,MAX3_KERNEL_y,MAX3_KERNEL_x,MAX3_PAD_y,MAX3_PAD_x,MAX3_ACT_min,MAX3_ACT_max,MAX3_CHANNEL_in,CONV3_OUT,NULL,CONV3_MAX_OUT);
	//FC1				
	
	arm_fully_connected_q7(CONV3_MAX_OUT,wf_1,IP1_IN_DIM,IP1_OUT_DIM,IP1_BIAS_LSHIFT,IP1_OUT_RSHIFT,bf_1,FC1_OUT,buffer3);	

		max_fout=FC1_OUT[0];
	loc=0;
	
	for(int i=0; i<12;i++){
	if(FC1_OUT[i]>max_fout)
	{
		max_fout=FC1_OUT[i];
		loc=i;
	}
	}
	GPIOD->ODR=(val[loc]<<9);
	

}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 64;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV8;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_ADC1_Init(void)
{

  ADC_ChannelConfTypeDef sConfig = {0};

  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV8;
  hadc1.Init.Resolution = ADC_RESOLUTION_8B;
  hadc1.Init.ScanConvMode = DISABLE;
  hadc1.Init.ContinuousConvMode = ENABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DMAContinuousRequests = ENABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  sConfig.Channel = ADC_CHANNEL_0;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_112CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

}

/** 
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void) 
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream0_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);

}


static void MX_GPIO_Init(void)
{
GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, GPIO_PIN_9|GPIO_PIN_10|GPIO_PIN_11|GPIO_PIN_12 
                          |GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15, GPIO_PIN_SET);

  /*Configure GPIO pin : PE4 */
  GPIO_InitStruct.Pin = GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pin : PA0 */
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PD9 PD10 PD11 PD12 
                           PD13 PD14 PD15 */
  GPIO_InitStruct.Pin = GPIO_PIN_9|GPIO_PIN_10|GPIO_PIN_11|GPIO_PIN_12 
                          |GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI4_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI4_IRQn);
}

void Error_Handler(void)
{
  
}
