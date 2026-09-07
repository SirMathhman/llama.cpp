import { describe, expect, it } from 'vitest';
import { FileTypeCategory } from '$lib/enums';
import { getFileTypeCategory, getFileTypeCategoryByExtension } from '$lib/utils/file-type';

describe('iPhone .mov support', () => {
	it('detects video/quicktime as video by MIME type', () => {
		expect(getFileTypeCategory('video/quicktime')).toBe(FileTypeCategory.VIDEO);
	});

	it('detects .mov as video by extension when MIME is missing', () => {
		expect(getFileTypeCategoryByExtension('IMG_1234.MOV')).toBe(FileTypeCategory.VIDEO);
		expect(getFileTypeCategoryByExtension('clip.mov')).toBe(FileTypeCategory.VIDEO);
	});

	it('still detects the pre-existing video types', () => {
		expect(getFileTypeCategory('video/mp4')).toBe(FileTypeCategory.VIDEO);
		expect(getFileTypeCategory('video/ogg')).toBe(FileTypeCategory.VIDEO);
		expect(getFileTypeCategoryByExtension('clip.mp4')).toBe(FileTypeCategory.VIDEO);
	});

	it('does not misclassify audio or text', () => {
		expect(getFileTypeCategory('audio/wav')).toBe(FileTypeCategory.AUDIO);
		expect(getFileTypeCategoryByExtension('notes.md')).toBe(FileTypeCategory.TEXT);
	});
});
