import datetime
import xlsxwriter
import os

def generate_excel(records):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_path =  BASE_DIR + '/static/media/excel/' +current_time+'.xlsx'
    print(save_path)
    wbk = xlsxwriter.Workbook(save_path)
    sheet = wbk.add_worksheet('检测结论')
    sheet.write(0,0,'文件名')
    sheet.write(0,1,'检测结论')
    for i in range(len(records)):
        for j in range(len(records[i])):
            sheet.write(i+1,j,records[i][j])

    wbk.close()
    save_excel_path = current_time
    return save_excel_path


def download(request):
    from django.http import StreamingHttpResponse
    def file_iterator(file_name,chunk_size=512):
        with open(file_name,'rb') as f:
            while True:
                c = f.read(chunk_size)
                if c:
                    yield c
                else:
                    break

    save_excel_path = request.GET.get('download_url')
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    the_file_name = save_excel_path +'.xlsx'
    save_excel_path = BASE_DIR + '/static/media/excel/' + save_excel_path +'.xlsx'
    response = StreamingHttpResponse(file_iterator(save_excel_path))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(the_file_name)

    return response

