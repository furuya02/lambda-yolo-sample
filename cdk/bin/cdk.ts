#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { YoloSampleStack } from '../lib/cdk-stack';

const app = new cdk.App();
new YoloSampleStack(app, 'YoloSampleStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'ap-northeast-1'
  },
  description: 'Simple YOLO Object Detection Sample with Lambda Container',
});
